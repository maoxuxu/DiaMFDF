from torch import nn
import torch.nn.functional as F
import torch
import copy
import math

from src.Roberta import InteractionAttention

"""
class GraphConvLayer(nn.Module):
    \""" A GCN module operated on dependency graphs. \"""

    def __init__(self, opt, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_ouputs = torch.cat(output_list, dim=2)
        gcn_ouputs = gcn_ouputs + gcn_inputs

        out = self.Linear(gcn_ouputs)

        return out
"""

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, d_out=None):
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None: d_out = d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class InteractLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, config=None):
        super(InteractLayer, self).__init__()
        head_size = int(d_model / num_heads)
        self.config = config
        self.interactionAttention = InteractionAttention(num_heads, d_model, head_size, head_size, dropout, config=config)

        self.layer_norm_pre = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.layer_norm_post = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, global_x, mask, sentence_length,):
        x = self.layer_norm_pre(self.interactionAttention(x, global_x, mask,)[0] + x)
        x = self.layer_norm_post(self.ffn(x) + x)
        x = self.dropout(x)
        return x


class GraphConvLayer(nn.Module):
    def __init__(self, config, layer_num, input_dim, hidden_dim, output_dim, dropout):
        super(GraphConvLayer, self).__init__()
        self.config = config
        self.layer_list = nn.ModuleList()
        for i in range(layer_num):
            if i == layer_num - 1:
                self.layer_list.append(nn.Linear(hidden_dim, output_dim))
            elif i == 0:
                self.layer_list.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layer_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.gnn_dropout = nn.Dropout(dropout)
        self.gnn_activation = F.gelu

    def forward(self, x, mask, adj):
        D_hat = torch.diag_embed(torch.pow(torch.sum(adj, dim=-1), -1))
        if torch.isinf(D_hat).any():
            D_hat[torch.isinf(D_hat)] = 0.0
        adj = torch.matmul(D_hat, adj)
        # adj = torch.matmul(adj, D_hat)

        x_mask = mask.unsqueeze(-1)#.expand(-1, -1, x.size(-1))
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                x = self.gnn_dropout(x)
            x = torch.matmul(x, layer.weight.T) + layer.bias
            x = torch.matmul(adj, x)
            x = x * x_mask
            x = self.gnn_activation(x)
        return x


class MultiGraphConvLayer(nn.Module):
    def __init__(self, opt, input_dim, output_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = output_dim 
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(input_dim + self.head_dim * j, self.head_dim))
        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):
        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]

          
            # denom = adj.sum(dim=-1).unsqueeze(-1) + 1
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs
            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)
        return out


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn