import torch
import torch.nn as nn
import dgl.nn as dglnn
import dgl
from fast_gtn import FastGTNs
from dgl.nn.pytorch import GATConv
from decoders import InnerProductDecoder

class Node_Embedding(nn.Module):
    """HeteroGCN layer"""

    def __init__(self, in_feats, out_feats, dropout, rel_names):
        super().__init__()
        HeteroGraphdict = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(in_feats, out_feats)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict[rel] = graphconv
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = dglnn.HeteroGraphConv(HeteroGraphdict, aggregate='sum')
        self.bn_layer = nn.BatchNorm1d(out_feats)
        self.prelu = nn.PReLU()

    def forward(self, graph, inputs, bn=False, dp=False):
        h = self.embedding(graph, inputs)
        if bn and dp:
            h = {k: self.prelu(self.dropout(self.bn_layer(v))) for k, v in h.items()}
        elif dp:
            h = {k: self.prelu(self.dropout(v)) for k, v in h.items()}
        elif bn:
            h = {k: self.prelu(self.bn_layer(v)) for k, v in h.items()}
        else:
            h = {k: self.prelu(v) for k, v in h.items()}
        return h


class SemanticAttention(nn.Module):
    """The base attention mechanism used in layer attention."""

    def __init__(self, in_feats, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class Graph_attention(nn.Module):

    def __init__(self, in_feats, out_feats, num_heads, dropout):
        super().__init__()
        self.gat = GATConv(in_feats, out_feats, num_heads,
                                 dropout, dropout,
                                 activation=nn.PReLU(),
                                 allow_zero_in_degree=True)
        self.gat.reset_parameters()
        self.linear = nn.Linear(in_feats * num_heads, out_feats)
        self.prelu = nn.PReLU()
        self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, graph, inputs, bn=False):
        num_dis = graph.num_nodes('disease')
        num_drug = graph.num_nodes('drug')
        new_g = dgl.to_homogeneous(graph)
        new_h = torch.cat([i for i in inputs.values()], dim=0)
        new_h = self.gat(new_g, new_h)
        new_h = self.prelu(torch.mean(new_h, dim=1))
        if bn:
            return self.bn_layer(new_h[:num_dis]), self.bn_layer(new_h[num_dis:num_drug + num_dis])
        return new_h[:num_dis], new_h[num_dis:num_drug + num_dis]


class Model(nn.Module):
    """The overall MRDDA architecture."""

    def __init__(self, etypes, ntypes, in_feats, hidden_feats, num_heads, dropout, num_nodes, use_gtn=False,
                 gtn_channels=1, gtn_layers=1):
        super(Model, self).__init__()
        self.ntypes = ntypes
        if 'drug' in ntypes:
            self.drug_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.drug_linear.weight)
        if 'disease' in ntypes:
            self.disease_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.disease_linear.weight)
        if 'protein' in ntypes:
            self.protein_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.protein_linear.weight)
        if 'gene' in ntypes:
            self.gene_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.gene_linear.weight)
        if 'pathway' in ntypes:
            self.pathway_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.pathway_linear.weight)

        self.HeteroGCN_layer1 = Node_Embedding(hidden_feats, hidden_feats, dropout, etypes)
        self.HeteroGCN_layer2 = Node_Embedding(hidden_feats, hidden_feats, dropout, etypes)
        self.gat_layer = Graph_attention(hidden_feats, hidden_feats, num_heads, dropout)
        self.layer_attention_drug = SemanticAttention(hidden_feats)
        self.layer_attention_dis = SemanticAttention(hidden_feats)
        self.predict = InnerProductDecoder(hidden_feats)
        
        # optional in-model GTN to learn node embeddings from graph structure
        self.hidden_feats = hidden_feats
        # Instantiate GTN/FastGTN immediately so state_dict works
        self.gtn_C = min(gtn_channels, len(etypes)) if len(etypes) > 0 else gtn_channels
        self.gtn_L = gtn_layers
        
        self.gtn = None # Will be overwritten below
        num_edge = len(etypes)
        num_channels = self.gtn_C
        w_in = self.hidden_feats
        w_out = self.hidden_feats
        
        class FastGTNArgs:
            pass
        fg_args = FastGTNArgs()
        fg_args.num_channels = num_channels
        fg_args.num_layers = self.gtn_L
        fg_args.node_dim = w_out
        fg_args.non_local = False
        fg_args.beta = 0.5
        fg_args.channel_agg = 'mean'
        fg_args.remove_self_loops = False
        fg_args.non_local_weight = 0
        
        fg_args.num_FastGTN_layers = 1
        fg_args.dataset = 'None'
        
        self.gtn = FastGTNs(num_edge_type=num_edge,
                            w_in=w_in,
                            num_class=w_out, 
                            num_nodes=num_nodes,
                            args=fg_args)
        self.gtn_proj = None # Not used/needed for FastGTN as it projects internally

    def gtn_embeddings(self , h , g) :
        
        # If GTN mode is enabled and external embeddings not provided, compute them here
        device = h[next(iter(h))].device

        # Build adjacency list expected by GTN: list of (edge_index, edge_value)
        g_homo = dgl.to_homogeneous(g)
        N = g_homo.num_nodes()
        u_all, v_all = g_homo.edges()
        e_types = g_homo.edata[dgl.ETYPE]
        A = []
        for i, etype in enumerate(g.etypes):
            mask = (e_types == i)
            u = u_all[mask].to(device)
            v = v_all[mask].to(device)
            if u.numel() == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_value = torch.empty((0,), dtype=torch.float32, device=device)
            else:
                edge_index = torch.stack([u, v], dim=0)
                edge_value = torch.ones(u.size(0), dtype=torch.float32, device=device)
            A.append((edge_index, edge_value))

        # GTN instantiation moved to __init__
        
        # build homogeneous feature tensor in the DGL node-type order and run GTN
        X_homo = torch.cat([h[nt] for nt in g.ntypes], dim=0)
        
        # FastGTN forward: A, X, num_nodes
        # FastGTN expects A as list of (edge_index, edge_value)
        H_gtn, _ = self.gtn(A, X_homo, num_nodes=N)
        # FastGTN already projects to hidden_feats if configured correctly (node_dim=hidden_feats, channel_agg='mean')
        H_proj = H_gtn

        # split according to node counts per type
        offsets = []
        cur = 0
        counts = [g.num_nodes(nt) for nt in g.ntypes]
        for c in counts:
            offsets.append((cur, cur + c))
            cur += c
        # find indices for drug and disease within g.ntypes
        drug_idx = g.ntypes.index('drug')
        dis_idx = g.ntypes.index('disease')
        ds, de = offsets[drug_idx]
        rs, re = offsets[dis_idx]
        mdrug = H_proj[ds:de]
        mdis = H_proj[rs:re]
        return mdrug , mdis
        
    def forward(self, g, x):
        # Apply initial per-type linear projections first
        h = {}
        for ntype in self.ntypes:
            h[ntype] = x[ntype]
        h['drug'] = self.drug_linear(h['drug'])
        h['disease'] = self.disease_linear(h['disease'])
        if 'protein' in self.ntypes:
            h['protein'] = self.protein_linear(h['protein'])
        if 'gene' in self.ntypes:
            h['gene'] = self.gene_linear(h['gene'])
        if 'pathway' in self.ntypes:
            h['pathway'] = self.pathway_linear(h['pathway'])

        drug_emb_list, dis_emb_list = [], []
        
        mdrug , mdis = self.gtn_embeddings(h , g)
        drug_emb_list.append(mdrug)
        dis_emb_list.append(mdis)

        drug_emb_list.append(h['drug'])
        dis_emb_list.append(h['disease'])
        
        h = self.HeteroGCN_layer1(g, h, bn=True, dp=True)
        h = self.HeteroGCN_layer2(g, h, bn=True, dp=True)
        drug_emb_list.append(h['drug'])
        dis_emb_list.append(h['disease'])
        
        h['disease'], h['drug'] = self.gat_layer(g, h)
        drug_emb_list.append(h['drug'])
        dis_emb_list.append(h['disease'])
        
        h['drug'] = self.layer_attention_drug(torch.stack(drug_emb_list, dim=1))
        h['disease'] = self.layer_attention_dis(torch.stack(dis_emb_list, dim=1))

        return self.predict(h)
