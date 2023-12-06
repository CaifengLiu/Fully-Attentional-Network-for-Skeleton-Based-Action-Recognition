import torch
import torch.nn as nn
import math
import numpy as np


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=32,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=True, att_t=True, attentiondrop=0, use_pes=True, use_pet=True):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        atts = torch.zeros((1, num_subset, num_node, num_node))
        self.register_buffer('atts', atts)
        self.ff_nets = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                            requires_grad=True)

        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        attt = torch.zeros((1, num_subset, num_frame, num_frame))
        self.register_buffer('attt', attt)
        self.ff_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
        self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame),
                                            requires_grad=True)
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()
        attention = self.atts
        y = x
        q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                               dim=1)  # nctv -> n num_subset c'tv
        attention = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
        attention = attention + self.attention0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
            .view(N, self.num_subset * self.in_channels, T, V)
        y = self.out_nets(y)  # nctv

        y = self.relu(self.downs1(x) + y)

        y = self.ff_nets(y)

        y = self.relu(self.downs2(x) + y)


        attention = self.attt
        z = y
        q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                               dim=1)  # nctv -> n num_subset c'tv
        attention = attention + self.tan(
                torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
        attention = attention + self.attention0t.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        z = torch.einsum('nctv,nstq->nscqv', [y, attention]).contiguous() \
            .view(N, self.num_subset * self.out_channels, T, V)
        z = self.out_nett(z)  # nctv

        z = self.relu(self.downt1(y) + z)

        z = self.ff_nett(z)

        z = self.relu(self.downt2(y) + z)

        return z

class FAN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset=3, dropout=0., config=None, num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True, liftpool=False):
        super().__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        param = {
            'num_node': num_point,
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                    **param))
            num_frame = int(num_frame / stride + 0.5)

        self.fc = nn.Linear(self.out_channels, num_class)

        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        """

        :param x: N M C T V
        :return: classes scores
        """
        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = self.input_map(x)

        for i, m in enumerate(self.graph_layers):
                x = m(x)

        # NM, C, T, V
        x = x.view(N, M, self.out_channels, -1) #N, M, C, T*V
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial, [N, M, T*V, C] -> [N, M*T*V, C, 1]
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)  # whole spatial of one channel

        return self.fc(x)

