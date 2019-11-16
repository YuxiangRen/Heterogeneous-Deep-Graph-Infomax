import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import NodeAttentionLayer, SemanticAttentionLayer


class HGAT(nn.Module):
    def __init__(self, nfeat, nhid, shid, alpha, nheads, P):
        """Dense version of GAT."""
        super(HGAT, self).__init__()
        self.node_level_attentions = []
        self.P = P #the number of meta-path
        for _ in range(P):
            self.node_level_attentions.append([NodeAttentionLayer(nfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)])

        for i, node_attentions_path in enumerate(self.node_level_attentions):
            for j, node_attention in enumerate(node_attentions_path):
                self.add_module('attention_path_{}_head_{}'.format(i,j), node_attention)

        self.semantic_level_attention = SemanticAttentionLayer(nhid*nheads, shid)
        
    def forward(self, x, adjs):
        x = torch.squeeze(x, 0)
#        print(x.size())
        meta_path_x = []
        for i, adj in enumerate(adjs):
            adj = torch.squeeze(adj, 0)
            m_x = torch.cat([att(x, adj) for att in self.node_level_attentions[i]], dim=1)
#            print(m_x.size())
            meta_path_x.append(m_x)
        
        x = torch.cat([m_x for m_x in meta_path_x], dim=0)
#        print(x.size())
        x = self.semantic_level_attention(x, self.P)
        
        x = torch.unsqueeze(x, 0) 
        return x