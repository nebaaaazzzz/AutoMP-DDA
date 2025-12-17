import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce
from torch.nn import Parameter
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.utils import add_self_loops

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def kaiming_uniform(tensor, fan, a):
    bound = math.sqrt(6 / ((1 + a**2) * fan))
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)

class GCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 args=None):
        super(GCNConv, self).__init__('add', flow='target_to_source')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels,out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None, args={'remove_self_loops':False}):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        loop_weight = torch.full((num_nodes, ),
                                 0,
                                dtype=edge_weight.dtype,
                                device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype, args=self.args)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_nodes, num_layers):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=False))
        self.layers = nn.ModuleList(layers)
        
        # Optimization: Explicitly set cached=False for GCN to save memory
        self.gcn = GCNConv(in_channels=self.w_in, out_channels=w_out, cached=False)
    def normalization(self, H, num_nodes):
        norm_H = []
        for i in range(self.num_channels):
            edge, value = H[i]
            deg_row, deg_col = self.norm(edge.detach(), num_nodes, value)
            value = (deg_row) * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, A, X, num_nodes=None, eval=False):
        if num_nodes is None:
            num_nodes = self.num_nodes
            
        for i in range(self.num_layers):
            if i == 0:
                H, _ = self.layers[i](A, num_nodes, eval=eval)
            else:                
                H, _ = self.layers[i](A, num_nodes, H, eval=eval)
            
            # Normalization
            H = self.normalization(H, num_nodes)
            
        for i in range(self.num_channels):
            edge_index, edge_weight = H[i][0], H[i][1]
            
            # Detach edge_index to prevent gradient tracking on structure if not needed for GCN
            # (GCN typically only needs grad on X and weights, not indices)
            if i == 0:                
                X_ = self.gcn(X, edge_index=edge_index.detach(), edge_weight=edge_weight)
                X_ = F.relu(X_)
            else:
                X_tmp = F.relu(self.gcn(X, edge_index=edge_index.detach(), edge_weight=edge_weight))
                X_ = torch.cat((X_, X_tmp), dim=1)

        return X_ 
    
class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, first=True, keep_rate=1, threshold=0.05):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        self.keep_rate = keep_rate
        self.threshold = threshold
        
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
    
    def forward(self, A, num_nodes, H_=None, eval=False):
        if self.first:
            result_A = self.conv1(A, num_nodes, eval=eval)
            result_B = self.conv2(A, num_nodes, eval=eval)                
            W = [(F.softmax(self.conv1.weight, dim=1)), (F.softmax(self.conv2.weight, dim=1))]
        else:
            result_A = H_
            result_B = self.conv1(A, num_nodes, eval=eval)
            W = [(F.softmax(self.conv1.weight, dim=1))]
        
        H = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]
            
            # 1. Capture original device (likely GPU)
            original_device = a_edge.device

            # 2. Move inputs to CPU to avoid GPU OOM during expansion
            a_edge_cpu = a_edge.cpu()
            a_value_cpu = a_value.cpu()
            b_edge_cpu = b_edge.cpu()
            b_value_cpu = b_value.cpu()

            # 3. Perform Matrix Multiplication on CPU (RAM)
            # This is where the graph explodes in size safely
            edge_index_cpu, value_cpu = spspmm(
                a_edge_cpu, a_value_cpu, 
                b_edge_cpu, b_value_cpu, 
                num_nodes, num_nodes, num_nodes
            )
            
            # 4. Threshold Pruning on CPU
            # We reduce the size BEFORE moving back to GPU
            mask = value_cpu > self.threshold
            edge_index_cpu = edge_index_cpu[:, mask]
            value_cpu = value_cpu[mask]

            # 5. Top-K Pruning on CPU
            if self.keep_rate < 1.0:
                num_edges = value_cpu.numel()
                k = int(num_edges * self.keep_rate)
                
                if k < num_edges and k > 0:
                    topk_values, topk_indices = torch.topk(value_cpu, k)
                    edge_index_cpu = edge_index_cpu[:, topk_indices]
                    value_cpu = topk_values

            # 6. Move the now-small tensors back to the original device (GPU)
            edge_index = edge_index_cpu.to(original_device)
            value = value_cpu.to(original_device)
            
            H.append((edge_index, value))
            
        return H, W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = None
        self.num_nodes = num_nodes
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A, num_nodes, eval=eval):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            edges_list = []
            values_list = []
            
            for j, (edge_index, edge_value) in enumerate(A):
                # OPTIMIZATION: Skip unnecessary concatenations for low weights
                w = filter[i][j]
                if w < 1e-4: 
                    continue
                
                edges_list.append(edge_index)
                values_list.append(edge_value * w)
            
            # If all weights were too small, return empty
            if not edges_list:
                results.append((torch.empty((2,0), device=A[0][0].device, dtype=torch.long), 
                                torch.empty(0, device=A[0][0].device)))
                continue

            total_edge_index = torch.cat(edges_list, dim=1)
            total_edge_value = torch.cat(values_list)
            
            # Coalesce: sum up duplicate edges
            index, value = coalesce(total_edge_index, total_edge_value, m=num_nodes, n=num_nodes, op='add')
            results.append((index, value))
            
        return results