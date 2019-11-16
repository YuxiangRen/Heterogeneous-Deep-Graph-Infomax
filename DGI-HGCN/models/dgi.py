import torch
import torch.nn as nn
from layers import HGCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, nfeat, nhid, shid, P, act):
        super(DGI, self).__init__()
        self.hgcn = HGCN(nfeat, nhid, shid, P, act)
        
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(nhid)

    def forward(self, seq1, seq2, adjs, sparse, msk, samp_bias1, samp_bias2):
        
        h_1 = self.hgcn(seq1, adjs, sparse)

        c = self.read(h_1, msk)
        
        c = self.sigm(c)

        h_2 = self.hgcn(seq2, adjs, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h_1 = self.hgcn(seq, adjs, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

