import torch
import torch.nn as nn
import torch.nn.functional as F

class HAGNN_Layer(nn.Module):
    # 单个注意力头
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, threshold=0.01):
        super(HAGNN_Layer, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.threshold = threshold

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # attention
        self.a_src = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a_dst = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj, hub_mask):
        Wh = torch.mm(h, self.W) 
        N = Wh.size(0)

        # 计算attention
        e_src = torch.mm(Wh, self.a_src) # (N, 1)
        e_dst = torch.mm(Wh, self.a_dst) # (1, N)
        e = self.leakyrelu(e_src + e_dst.T) # (N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        
        # hub_mask=1，减去 T ；hub_mask=0，减去 0
        threshold_matrix = hub_mask * self.threshold
        attention = F.relu(attention - threshold_matrix)
        
        # 聚合
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class HAGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, threshold):
        super(HAGNN, self).__init__()
        self.dropout = dropout

        # 多头拼接 
        self.hidden_heads = nn.ModuleList([
            HAGNN_Layer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, threshold=threshold) 
            for _ in range(nheads)
        ])

        # 多头平均 
        self.out_heads = nn.ModuleList([
            HAGNN_Layer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False, threshold=threshold)
            for _ in range(nheads)
        ])

    def forward(self, x, adj, hub_mask):
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = torch.cat([head(x, adj, hub_mask) for head in self.hidden_heads], dim=1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        
        head_outputs = [head(x, adj, hub_mask) for head in self.out_heads]
        x = torch.stack(head_outputs, dim=2).mean(dim=2)
        
        return F.log_softmax(x, dim=1)
