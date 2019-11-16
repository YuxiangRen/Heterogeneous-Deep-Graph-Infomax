import torch
import torch.nn as nn
from layers import AvgReadout, Discriminator, HGAT
#from layers.gat import SpGAT, GATß
 
class DGI(nn.Module):
    def __init__(self, nfeat, nhid, shid, alpha, nheads, P):
        super(DGI, self).__init__()
        self.hgat = HGAT(nfeat, nhid, shid, alpha, nheads, P) 
     
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()
        
        self.relu = nn.ReLU()
        self.disc = Discriminator(nhid*nheads)
#        self.disc = Discriminator(nhid)
    def forward(self, seq1, seq2, adjs, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.hgat(seq1, adjs)

        c = self.read(h_1, msk)
        
        c = self.sigm(c)
#        print(c)
        h_2 = self.hgat(seq2, adjs)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h_1 = self.hgat(seq, adjs)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

