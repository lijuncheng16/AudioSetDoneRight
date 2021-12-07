import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torch.autograd import Variable as V


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, freeze=False, bb_type='res34', transformer=False):
        super(Net, self).__init__()
        if bb_type == 'res34':
            model = models.resnet34(pretrained=True)
        elif bb_type == 'res152':
            model = models.resnet152(pretrained=True)
        else:
            assert(0)
        self.video_bb = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        if freeze:
            for param in self.video_bb.parameters():
                param.requires_grad = False
#         if transformer:
#             self.trans = nn.ModuleList([Transformer(hidden_dim=n_hidden, heads=4, dropout=0.0) for i in range(2)])
        self.transformer = transformer
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        self.drop =torch.nn.Dropout(p=0.0)
        self.act = torch.nn.ReLU()
        
    # B: batch size
    # T: temporal (5 frames)
    # C: RGB channels (3 channels)
    # W: width
    # H: height
    # E: emb dim
    def forward(self, x):
        # x: BxTxCxWxH
        b,t,c,w,h = x.shape
        x = x.reshape(b*t,c,w,h) # BTxCxWxH
        x = self.video_bb(x) # BTxE

        if self.transformer:
            # mapping
            x = self.hidden1(x) # BTxE
            # recover
            x = x.reshape(b,t,-1) # BxTxE
            mask = torch.ones(b,5).bool().cuda()
            for i in range(2):
                x = self.trans[i](x, mask)
            x = x[:,0,:]
            #x = x.mean(dim=1)
        else:
            # recover
            x = x.reshape(b,t,-1) # BxTxE
            # pooling 
            x = torch.mean(x, dim=1) # BxE
            #x = self.drop(x)
            x = self.act(self.drop(self.hidden1(x)))
            x = self.act(self.drop(self.hidden2(x))) + x # with residual
        x = self.predict(x)  
        return x

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1])


class Transformer(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.mha=MultiHeadAttention(heads, hidden_dim, dp=dropout)
        self.norm1=nn.LayerNorm(hidden_dim, eps=1e-05)
        self.ffn=FFN(hidden_dim, 2048, hidden_dim)
        self.norm2=nn.LayerNorm(hidden_dim, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # MHA
        attns = self.mha(x, mask)
        attn = attns[0]
        #attn = self.dropout(attn)
        x = x + attn
        #x = self.norm1(x)
        ## FFN
        x = x + self.ffn(x)
        #x = self.norm2(x)
        x *= mask.unsqueeze(-1).to(x.dtype)
        return x

class FFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = F.gelu #if config.gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dp=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dp
        #assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, n_heads*dim)
        self.k_lin = nn.Linear(dim, n_heads*dim)
        self.v_lin = nn.Linear(dim, n_heads*dim)
        self.out_lin = nn.Linear(n_heads*dim, dim)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen
        else:
            klen = kv.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        #dim_per_head = self.dim // n_heads
        dim_per_head = self.dim
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None: # self attention
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None: # or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        #weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        return outputs

