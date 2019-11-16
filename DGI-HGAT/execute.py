import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import argparse
import os
import glob
#import time
import random

from models import DGI, LogReg
from utils import process

#dataset = 'cora'
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--shid', type=int, default=8, help='Number of semantic level hidden units.')
parser.add_argument('--out', type=int, default=8, help='Number of output feature dimension.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=30, help='Patience')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if torch.cuda.is_available():
 device = torch.device("cuda")
 torch.cuda.set_device(3)
else:
 device = torch.device("cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
    
# training params
batch_size = 1
nb_epochs = args.epochs
patience = args.patience
lr = args.lr
l2_coef = args.weight_decay
drop_prob = args.dropout
hid_units = args.hidden 
shid = args.shid
sparse = args.sparse
nout = hid_units * args.nb_heads
exp = 7
#nout = hid_units
#nonlinearity = 'prelu' # special name to separate parameters

adjs, features, labels, idx_train, idx_val, idx_test = process.load_data()

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]


P=int(len(adjs))

nor_adjs = []
for adj in adjs:
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = adj[np.newaxis]
    nor_adjs.append(adj)
nor_adjs = torch.FloatTensor(np.array(nor_adjs))

#
#if sparse:
#    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
#else:
#    adj = (adj + sp.eye(adj.shape[0])).todense()

#features = torch.FloatTensor(features[np.newaxis])
#features = torch.FloatTensor(features)
##if not sparse:
#adj = torch.FloatTensor(adj)
#labels = torch.FloatTensor(labels[np.newaxis])
features, _ = process.preprocess_features(features)
features = torch.FloatTensor(features[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])


idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, shid, args.alpha, args.nb_heads, P) 
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
#msk = torch.randn(1, nb_nodes).ge(0.5).float() 
#if torch.cuda.is_available():
if args.cuda:
    print('Using CUDA')
    model.cuda()
    
    features = features.cuda()

    nor_adjs = nor_adjs.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
#    msk = msk.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

#    if torch.cuda.is_available():
    if args.cuda:
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    logits = model(features, shuf_fts, nor_adjs, sparse, None, None, None) 

    loss = b_xent(logits, lbl)

    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), str('best_dgi_head_'+str(args.nb_heads)+'_nhidden_'+str(args.hidden)+'_exp_'+str(exp)+'.pkl'))
    else:
        cnt_wait += 1
    print("wait: "+str(cnt_wait))
    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))


model.load_state_dict(torch.load(str('best_dgi_head_'+str(args.nb_heads)+'_nhidden_'+str(args.hidden)+'_exp_'+str(exp)+'.pkl')))

embeds, _ = model.embed(features, nor_adjs, sparse, None)
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

tot = torch.zeros(1)
if args.cuda:
   tot = tot.cuda()
tot_mac = 0
accs = []
mac_f1 = []

for _ in range(5):
    bad_counter = 0
    best = 10000
    loss_values = []
    best_epoch = 0
    train_patience = 50
    
    log = LogReg(nout, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    if args.cuda:
       log.cuda()

    for epoch in range(100000):
        log.train()
        opt.zero_grad()
        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        logits_val = log(val_embs)
        loss_val = xent(logits_val, val_lbls)
        loss_values.append(loss_val)
#        print("train_loss: "+ str(loss) +"  "+"val_loss: "+ str(loss_val) )
        loss.backward()
        opt.step()
        torch.save(log.state_dict(), '{}.mlp.pkl'.format(epoch))
        if loss_values[-1] < best:
           best = loss_values[-1]
           best_epoch = epoch
           bad_counter = 0
        else:
           bad_counter += 1
        
        if bad_counter == train_patience:
            break
        
        files = glob.glob('*.mlp.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
               os.remove(file)
        
        
    files = glob.glob('*.mlp.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    
    print("Optimization Finished!")  
    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    log.load_state_dict(torch.load('{}.mlp.pkl'.format(best_epoch)))
    
    files = glob.glob('*.mlp.pkl')
    for file in files:
            os.remove(file)
    
    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc)
#    print(acc)
#    tot += acc
    mac = torch.Tensor(np.array(process.macro_f1(preds, test_lbls))) 
#    tot_mac += mac
    mac_f1.append(mac)
#print('Average accuracy:', tot / 50)
#print('Average mac_f1:', tot_mac / 50)
accs = torch.stack(accs)
print('Average accuracy:',accs.mean())
print('accuracy std:',accs.std())
mac_f1 = torch.stack(mac_f1)
print('Average mac_f1:', mac_f1.mean())
print('mac_f1 std:',mac_f1.std())