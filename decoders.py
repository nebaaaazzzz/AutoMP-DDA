from torch import nn
class InnerProductDecoder(nn.Module):
    """Decoder layer for prediction"""

    def __init__(self, input_dim=None, dropout=0.4):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)
        self.linear1 = nn.Linear(128, input_dim)
        self.linear2 = nn.Linear(128, input_dim)

    def forward(self, feature):
        R = self.dropout(feature['drug'])
        D = self.dropout(feature['disease'])
        D = self.weights(D)
        outputs = R @ D.T
        return outputs
class CrossAttentionScoreDecoder(nn.Module):
    """
    Cross-attention decoder that outputs score matrix [R, D],
    same as inner-product/bilinear decoder.
    """

    def __init__(self, input_dim=128, dropout=0.4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)

        # Final vector to convert attended disease into a scalar
        self.score_vec = nn.Linear(input_dim, 1, bias=False)

    def forward(self, feature):
        R = self.dropout(feature["drug"])      # [R, F]
        D = self.dropout(feature["disease"])   # [D, F]

        # Q from drug, K from disease
        Q = self.W_q(R)       # [R, F]
        K = self.W_k(D)       # [D, F]

        # Cross-attention logits BEFORE softmax
        logits = Q @ K.T      # [R, D]

        # Optional: scale logits (Transformer style)
        logits = logits / (Q.shape[-1] ** 0.5)

        # Convert logits → final scores (real number for BCE)
        # Instead of V, we directly use logits as score matrix.
        return logits