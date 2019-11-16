import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
#        print(h_pl.size())
#        print(h_mi.size())
#        print(c.size())
        c_x = torch.unsqueeze(c, 1)
#        print(c_x.size())
        c_x = c_x.expand_as(h_pl)
#        print(c_x.size())
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
#        print(sc_1.size())
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
#        print(sc_2.size())
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
#        print(logits)
        return logits

