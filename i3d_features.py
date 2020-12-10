import re
import os, sys, codecs
import subprocess
import math
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pickle
from PIL import Image
from tqdm.notebook import tqdm
#import tqdm
import cv2
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

# take from i3d.py and feature_extract.py into one file


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe=nn.Parameter(pe)
        torch.nn.init.normal_(self.pe,std = d_model ** -0.5)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BERTEmbedding3(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding(d_model=input_dim,max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence)+sequence
        return self.dropout(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT2(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob


        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(self.clsToken, std = hidden ** -0.5)



        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding3(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                #nn.init.normal_(module.weight, mean=0, std=0.02)
                nn.init.uniform_(module.weight, -0.06, 0.06)
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x, sample


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)


    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')


    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits


    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


def _inception(model_path, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InceptionI3d(400, in_channels=3)
    if model_path=='':
        return model
    params = torch.load(model_path)
    model.load_state_dict(params)
    return model


class rgb_I3D64f(nn.Module):
    def __init__(self, modelPath='./weights/rgb_imagenet.pth'):
        super(rgb_I3D64f, self).__init__()
        # self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.features=nn.Sequential(*list(_inception(model_path=modelPath).children())[3:])

        # self.fc_action = nn.Linear(1024, num_classes)
        # for param in self.features.parameters():
        #     param.requires_grad = True

        # torch.nn.init.xavier_uniform_(self.fc_action.weight)
        # self.fc_action.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.dp(x)
        # x = self.fc_action(x)
        return x


class rgb_I3D64f_bert2(nn.Module):
    def __init__(self, modelPath='./weights/rgb_imagenet.pth'):
        super(rgb_I3D64f_bert2, self).__init__()
        self.hidden_size=1024
        self.n_layers=1
        self.attn_heads=8
        # self.num_classes=num_classes
        # self.length=length
        self.dp = nn.Dropout(p=0.8)


        self.features=nn.Sequential(*list(_inception(model_path=modelPath).children())[3:])

        for param in self.features.parameters():
            param.requires_grad = True


        self.bert = BERT2(self.hidden_size, 8 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        # self.fc_action = nn.Linear(self.hidden_size, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(8, 1, 1))

        # torch.nn.init.xavier_uniform_(self.fc_action.weight)
        # self.fc_action.bias.data.zero_()

    def forward(self, x):
        # print('input\t' + str(np.shape(x)))
        x = self.features(x)
        # print('feature extracted\t' + str(np.shape(x)))
        x = self.avgpool(x)
        # print('average pooled\t' + str(np.shape(x)))
        x = x.view(x.size(0), self.hidden_size,8)
        x = x.transpose(1,2)
        norm = x.norm(p=2, dim = -1, keepdim=True)
        x = x.div(norm)
        input_vectors=x
        # print('transformed\t' + str(np.shape(x)))
        output , maskSample = self.bert(x)
        # print('BERTed\t' + str(np.shape(output)))
        # classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        # output=self.dp(classificationOut)
        # print('BERTed2\t' + str(np.shape(output)))
        # x = self.fc_action(output)
        # print('output\t' + str(np.shape(x)))
        # print(np.shape(x))
        # return x, input_vectors, sequenceOut, maskSample
        return sequenceOut


class QRDataset(Dataset):
    def __init__(self, img_path, transform = None):
        self.img_path = img_path

        self.img_label = np.zeros(len(img_path))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, self.img_path[index]

    def __len__(self):
        return len(self.img_path)


def normalize(x, copy = False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1, -1), copy = copy))
        #return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
    else:
        return sknormalize(x, copy = copy)
        #return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]


def my_squeeze(arr):
    print(np.shape(arr))
    # arr = arr.squeeze(4)
    # arr = arr.squeeze(3)
    # arr = arr.squeeze(0)
    arr = arr.squeeze()
    print(np.shape(arr))
    return arr


class Img2Vec():
    def __init__(self, model='i3d', layer='default', layer_output_size=512):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model == "i3d":
            self.model = rgb_I3D64f().to(device)
        elif model == "vgg16":
            self.model = vgg.vgg16.to(device)
        elif  model == "bert":
            self.model = rgb_I3D64f_bert2().to(device)
            # take the bottom layer out
        else:
            print("What")
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def run(self, path):
        if not isinstance(path, list):
            path = [path]

        data_loader = torch.utils.data.DataLoader(QRDataset(path, self.transform), batch_size = 64,
                                                  shuffle = False, num_workers = 16)

        my_embedding = None

        with torch.no_grad():
            for index, batch_data in enumerate(data_loader):
                batch_data = batch_data[0].unsqueeze(0).transpose(1,2).to(device)
                # print(batch_data)
                # print("input shape:")
                # print(np.shape(batch_data))
                output = self.model(batch_data)
                # print("output shape:")
                # print(np.shape(output))
                if my_embedding is not None:
                	my_embedding = torch.cat((my_embedding, output), 2)
                else:
                    my_embedding = output
        # my_embedding = output
        # print(my_embedding)
        # print("Final output shape:\t")
        # print(np.shape(my_embedding))
        return my_embedding


def frame_extract(inpath, outpath, fps):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    cmd = "ffmpeg -loglevel quiet -stats -i " + inpath + " -r " + str(fps) + " " + outpath + "frame%04d.jpg"
    # print(cmd)
    os.system(cmd)


def feature_extract(model, inpath):
    q = []
    for img in os.listdir(inpath):
        q.append(inpath + img)
    q.sort(key = lambda x: x.lower())
    print("feature extract:\tFrame_num:\t" + str(len(q)))

    x = model.run(q)
    x = my_squeeze(x)
    x = normalize(x.cpu())
    return x


def vid2vec(model, path, temp_path="./temp/"):
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    # extract frame
    frame_extract(path, temp_path, 8)
    # feature_num = len(os.listdir(temp_path))
    # pass the path of said
    feature = feature_extract(model, temp_path)
    # print(np.shape(feature))
    feature_num = np.shape(feature)[1]
    # files = glob.glob("./temp/")
    shutil.rmtree(temp_path)
    return feature, feature_num


print(sys.argv)
script_name, weight_path, inpath, outpath = sys.argv

img2vec = Img2Vec(model="bert")
feature, feature_num = vid2vec(img2vec, inpath)
with open(outpath, 'wb') as pk_file:
    pickle.dump(feature, pk_file, protocol = 4)
