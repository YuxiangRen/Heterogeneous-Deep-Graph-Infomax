import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN, SemanticAttentionLayer


class HGCN(nn.Module):
    def __init__(self, nfeat, nhid, shid, P, act):
        """Dense version of GAT."""
        super(HGCN, self).__init__()
        self.gcn_level_embeddings = []
        self.P = P #number of meta-Path
        for _ in range(P):
            self.gcn_level_embeddings.append(GCN(nfeat, nhid, act, bias=True))
            
        for i, gcn_embedding_path in enumerate(self.gcn_level_embeddings):
            
                self.add_module('gcn_path_{}'.format(i), gcn_embedding_path)

        self.semantic_level_attention = SemanticAttentionLayer(nhid, shid)

        
    def forward(self, x, adjs, sparse):
        meta_path_x = []
        for i, adj in enumerate(adjs):
            m_x = self.gcn_level_embeddings[i](x, adj, sparse)
            meta_path_x.append(m_x)
        
        x = torch.cat([m_x for m_x in meta_path_x], dim=0)
        
        x = self.semantic_level_attention(x, self.P)
#        print(x.size())
        x = torch.unsqueeze(x, 0)   
#        print(x.size())
        return x


