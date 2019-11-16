
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(SemanticAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
#input (PN)*F 
    def forward(self, input, P):
        h = torch.mm(input, self.W)
        #h=(PN)*F'
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0],1))
        #h_prime=(PN)*F'
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P,-1)       
        #semantic_attentions = P*N
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1,keepdim=True)
        #semantic_attentions = P*1
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
        print(semantic_attentions)
        semantic_attentions = semantic_attentions.view(P,1,1)
        semantic_attentions = semantic_attentions.repeat(1,N,self.in_features)
#        print(semantic_attentions)
        #input_embedding = P*N*F
        input_embedding = input.view(P,N,self.in_features)
        
        #h_embedding = N*F
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()
        
        return h_embedding

