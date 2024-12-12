#  -*- coding: UTF-8 -*
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from utils.GCN_models import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputEmbedding(nn.Module):
    def __init__(self, input_shape, d_model=4):
        super().__init__()
        self.input_shape = input_shape
        self.d_model = d_model
        self.input_embedding = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_shape[2], self.d_model)),
            ('activation', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(self.d_model, self.d_model))
        ]))
    def forward(self, x):
        x = self.input_embedding(x)
        return x.transpose(2, 3)
class ScaleDotProductAttention_spatial(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        x = torch.matmul(attn, v)
        return x, attn
class MultiHeadAttention_spatial(nn.Module):
    def __init__(self,d_model, n_head, d_k, d_v, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.w_qs = nn.Linear(d_model * seq_len, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model * seq_len, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model * seq_len, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model * seq_len, bias=False)
        self.attention = ScaleDotProductAttention_spatial(temperature=d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model * seq_len, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        sz_b, n_q, n_k, n_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, n_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, n_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, n_v, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, n_q, -1)
        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)

        return q, attn


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        nn.Module
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
class Transformer_spatial(nn.Module):
    def __init__(self, gcn_flag, adj, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):
        super().__init__()
        self.gcn_flag = gcn_flag
        #self.adj = torch.tensor(adj, dtype=torch.float32)
        self.adj = torch.clone(adj).detach()
        #tensor.clone().detach()
        #b = torch.tensor(a)

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_inner = d_inner
        self.seq_len = seq_len
        self.attn = MultiHeadAttention_spatial(self.d_model, self.n_head, self.d_k, self.d_v, self.seq_len,
                                               dropout=dropout)
        self.ff = FeedForward(self.d_model * self.seq_len, self.d_inner, dropout=dropout)
        #增加权重系数与gcn
        self.norm_adj = nn.InstanceNorm2d(1)
        self.gcn = GCN(d_model, d_model * 2, d_model, dropout)
        self.fs = nn.Linear(d_model, d_model)
        self.fg = nn.Linear(d_model, d_model)
    def forward(self, x, v, mask=None):
        batch_size, seq_len, features, cells = x.shape
        # GCN 部分 X_G 400 0 64
        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)
        # 每个时间步计算gcn获取空间关系
        q = x.reshape(batch_size, cells, seq_len , features)

        x = x.reshape(batch_size, seq_len * features, cells).transpose(1, 2)
        v = v.reshape(batch_size, seq_len * features, cells).transpose(1, 2)
        x, attn = self.attn(x, x, v, mask=mask)
        x = self.ff(x)
        # 增加gcn
        X_G = torch.Tensor(0, cells, seq_len, features).to(device)

        for batch in range(q.shape[0]):
            X_g = torch.Tensor(q.shape[1], 0, q.shape[3]).to(device)
            for t in range(q.shape[2]):
                o = self.gcn(q[batch, :, t, :], self.adj)
                o = o.unsqueeze(1)  # shape [N, 1, C]
                X_g = torch.cat((X_g, o), dim=1).to(device)
            X_G = torch.cat((X_G, X_g.unsqueeze(0)), dim=0).to(device)
        g = torch.sigmoid(self.fs(x.view(batch_size, cells, seq_len, features)) + self.fg(X_G))  # (7)


        if self.gcn_flag == 0:
            x_final = x.transpose(1, 2).view(batch_size, seq_len, features, cells)
        elif self.gcn_flag == 1:
            x_final = X_G.view(batch_size, seq_len, features, cells)
        elif self.gcn_flag == 2: #加权
            x_final = g * x.view(batch_size, cells, seq_len, features) + (1 - g) * X_G
            x_final = x_final.view(batch_size, seq_len, features, cells)
        elif self.gcn_flag == 3:
            x_final = x.view(batch_size, cells, seq_len, features) + X_G
            x_final = x_final.view(batch_size, seq_len, features, cells)
        return x_final, attn
class ScaleDotProductAttention_temporal(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(3, 4))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        x = torch.matmul(attn, v)
        return x, attn
class MultiHeadAttention_temporal(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaleDotProductAttention_temporal(temperature=d_k ** 0.5, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    def forward(self, q, k, v, mask=None):
        sz_b, n_q, n_k, n_v, len_q, len_k, len_v = \
            q.size(0), q.size(1), k.size(1), v.size(1), q.size(2), k.size(2), v.size(2)
        residual = q
        q = self.w_qs(q).view(sz_b, n_q, len_q, self.n_head,
                              self.d_k)
        k = self.w_ks(k).view(sz_b, n_k, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, n_v, len_v, self.n_head, self.d_v)
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(2, 3).contiguous().view(sz_b, n_q, len_q, -1)
        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        return q, attn


class Transformer_temporal(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_inner = d_inner
        self.seq_len = seq_len
        self.attn = MultiHeadAttention_temporal(self.d_model, self.n_head, self.d_k, self.d_v, self.seq_len,
                                                dropout=dropout)
        self.ff = FeedForward(self.d_model, self.d_inner, dropout=dropout)
    def forward(self, x, v, mask=None):
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)
        x, attn = self.attn(x, x, v, mask=mask)
        x = self.ff(x)
        x = x.permute(0, 2, 3, 1)
        return x, attn
class Transformer_st(nn.Module):
    def __init__(self, gcn_flag, adj, n_layers, d_model, d_inner, nheads_spatial, nheads_temporal, dk_s, dv_s, dk_t, dv_t, seq_len,
                 dropout=0.1):
        super().__init__()
        self.adj = adj
        self.n_layers = n_layers
        self.d_model = d_model
        self.nheads_spatial = nheads_spatial
        self.nheads_temporal = nheads_temporal
        self.dk_s = dk_s
        self.dv_s = dv_s
        self.dk_t = dk_t
        self.dv_t = dv_t
        self.d_inner = d_inner
        self.seq_len = seq_len
        self.spatial_layers = nn.ModuleList([
            Transformer_spatial(gcn_flag, adj, d_model, d_inner, nheads_spatial, dk_s, dv_s, seq_len, dropout=dropout)
            for _ in range(self.n_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            Transformer_temporal(d_model, d_inner, nheads_temporal, dk_t, dv_t, seq_len, dropout=dropout)
            for _ in range(self.n_layers)
        ])

    def forward(self, x, mask=None, return_attns=True):
        attn_spatial_list = []
        attn_temporal_list = []
        enc_o = x
        for i in range(self.n_layers):
            x = enc_o
            enc_o, attn_s = self.spatial_layers[i](x, enc_o, mask)
            enc_o = enc_o + x
            enc_o, attn_t = self.temporal_layers[i](x, enc_o, mask)
            enc_o = enc_o + x
            attn_spatial_list.append(attn_s)
            attn_temporal_list.append(attn_t)
        if return_attns:
            return enc_o, attn_spatial_list, attn_temporal_list
        return enc_o
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = drop_rate

    def forward(self, input):
        new_features = super(_DenseLayer, self).forward(input.contiguous())
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([input, new_features], 1)
        # return new_features
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            # layer = _DenseLayer(num_input_features, growth_rate,
            #                     bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
class iLayer(nn.Module):
    def __init__(self):
        super(iLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(1))
    def forward(self, x):
        w = self.w.expand_as(x)
        return x * w
class SAGCN(nn.Module):
    def __init__(self, gcn_flag, A, input_shape, meta_shape, cross_shape, growth_rate=12, num_init_features=12, bn_size=4,
                 drop_rate=0.1, nb_flows=25, fusion=1, maps=1, d_model=18, d_inner=128, nheads_spatial=6,
                 nheads_temporal=4, layers=3, dk_s=20, dk_t=20,
                 flags_meta=0, flags_cross=0):
        super().__init__()
        self.adj = A
        self.input_shape = input_shape
        self.meta_shape = meta_shape
        self.cross_shape = cross_shape
        self.filters = num_init_features
        self.channels = nb_flows
        self.fusion = fusion
        self.maps = maps
        self.h, self.w = self.input_shape[-2], self.input_shape[-1]
        self.inner_shape = self.input_shape[:2] + (self.filters,) + self.input_shape[-2:]
        self.seq_len = self.input_shape[1]
        self.d_model = d_model
        self.d_inner = d_inner
        self.nheads_spatial = nheads_spatial
        self.nheads_temporal = nheads_temporal
        self.layers = layers
        self.dk_s = dk_s
        self.dv_s = dk_s
        self.dk_t = dk_t
        self.dv_t = dk_t
        self.inp_emb = InputEmbedding(self.input_shape, self.d_model)
        self.transformer = nn.Sequential()
        transformer = Transformer_st(gcn_flag, self.adj,self.layers, self.d_model, self.d_inner, self.nheads_spatial, self.nheads_temporal,
                                     self.dk_s, self.dv_s,
                                     self.dk_t, self.dv_t, self.seq_len, drop_rate, )
        self.transformer.add_module('transformer', transformer)
        self.trg_word_prj = nn.Linear(self.input_shape[1], 1, bias=False)
        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(18, self.seq_len, 1)
        # # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(25, 1, 1)
        self.relu = nn.ReLU()
        # Each denseblock
        self.features = nn.Sequential()
        if (flags_meta == 1) & (flags_cross == 1):
            num_features = self.d_model * 3
        elif (flags_meta == 1) & (flags_cross == 0):
            num_features = self.d_model * 2
        elif (flags_meta == 0) & (flags_cross == 1):
            num_features = self.d_model * 2
        elif (flags_meta == 0) & (flags_cross == 0):
            num_features = self.d_model
        block_config = [6, 6, 6]
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # num_features = growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)  # theta=0.5
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                # trans = _Transition(num_input_features=num_features, num_output_features=num_features )  # theta=0.5
                # self.features.add_module('transition%d' % (i + 1), trans)
                # num_features = num_features
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relulast', nn.ReLU(inplace=True)) # 原本的
        self.features.add_module('convlast', nn.Conv2d(num_features, nb_flows,
                                                       kernel_size=1, padding=0, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, out, meta=None, cross=None):
        batch_size, seq_len, features, row_of_cells, col_of_cells = out.shape
        out = out.view(batch_size, seq_len, features, row_of_cells * col_of_cells).transpose(2, 3)
        out = self.inp_emb(out)
        out, attn_spatial_list, attn_temporal_list = self.transformer(out)
        out = self.trg_word_prj(out.permute(0, 2, 3, 1)).squeeze(-1)
        out = out.view(batch_size, -1, row_of_cells, col_of_cells)
        out = self.features(out)
        # torch.sigmoid(self.conv2(out))        # out = torch.sigmoid(out)
        # out = self.relu(self.conv2(out))    # 1 1 400 64等号左边 out shape: [1, output_T_dim, N, C]
        # out = out.permute(0, 3, 2, 1)           # 1 64 400 1等号左边 out shape: [1, C, N, output_T_dim]
        # out = self.conv3(out)                   # 等号左边 out shape: [1, 1, N, output_T_dim]
        # out = out.squeeze(0).squeeze(0)
        return out
