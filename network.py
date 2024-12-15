import torch.nn as nn
from torch.nn.functional import normalize
import torch

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]

        # split embedding into self.heads pieces
        values = values.reshape(N, self.heads, self.head_dim)
        keys = keys.reshape(N, self.heads, self.head_dim)
        queries = query.reshape(N, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nhd,nhd->nh", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=-1)
        '''
        out = torch.einsum("nh,nhd->nd", [attention, values]).reshape(
            N, self.heads*self.head_dim
        )'''
        out = torch.mean(attention.unsqueeze(-1) * values, dim=1, keepdim=False)

        # out = torch.mean(attention.unsqueeze(-1) * values, dim=1, keepdim=False)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, layers, heads, forward_expansion):
        super(Encoder, self).__init__()
        self.trans = nn.Linear(input_dim, feature_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    feature_dim,
                    heads,
                    forward_expansion,
                )
                for _ in range(layers)]
        )

    def forward(self, out):
        out = self.trans(out)
        for layer in self.layers:
            out = layer(out, out, out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, forward_expansion
        )

    def forward(self, x, value, key):
        attention = self.attention(x, x, x)
        query = self.norm(attention + x)
        out = self.transformer_block(value, key, query)
        return out

'''
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim, layers, heads, forward_expansion):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [DecoderBlock(feature_dim, heads, forward_expansion)
             for _ in range(layers)]
        )

        self.fc_out = nn.Linear(feature_dim, input_dim)

    def forward(self, x, z):

        for layer in self.layers:
            x = layer(x, z, z)

        out = self.fc_out(x)
        return out'''


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),  # 改
            nn.ReLU(),
            nn.Linear(2000, 500),  # 改
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, views, input_size, feature_dim, layers, heads, forward_expansion):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.views = views
        for v in range(views):
            self.encoders.append(Encoder(input_size[v], feature_dim, layers, heads, forward_expansion))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, xs):
        zs = []
        xrs = []

        for v in range(self.views):
            x = xs[v]
            z = self.encoders[v](x)
            # print(z.shape)
            xr = self.decoders[v](z)
            zs.append(z)
            xrs.append(xr)
        return zs, xrs
