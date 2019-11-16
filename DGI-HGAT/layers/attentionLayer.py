import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha, concat=True):
        super(NodeAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
#        attention = F.dropout(attention, self.nd_dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
#            return F.elu(h_prime)
            return F.leaky_relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SemanticAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
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
        self.leakyrelu = nn.LeakyReLU()
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

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


