import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os
import glob
from models import DGI, LogReg
from utils import process, clustering

#dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 10000
patience = 50
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512 #output of the GCN dimension
shid = 16 #input dimension of semantic-level attention
sparse = False
nonlinearity = 'prelu' # special name to separate parameters

if torch.cuda.is_available():
 device = torch.device("cuda")
 torch.cuda.set_device(3)
else:
 device = torch.device("cpu")
 
adjs, features, labels, idx_train, idx_val, idx_test = process.load_data()
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
P=int(len(adjs))

nor_adjs = []
sp_nor_adjs = []
for adj in adjs:
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    
    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_nor_adjs.append(sp_adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()
        adj = adj[np.newaxis]
        nor_adjs.append(adj)
features = torch.FloatTensor(features[np.newaxis])
if sparse:
        sp_nor_adjs = torch.FloatTensor(np.array(sp_nor_adjs))
else:
        nor_adjs = torch.FloatTensor(np.array(nor_adjs))
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, shid, P, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_nor_adjs = sp_nor_adjs.cuda()
    else:
        nor_adjs = nor_adjs.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

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

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    logits = model(features, shuf_fts, sp_nor_adjs if sparse else nor_adjs, sparse, None, None, None) 

    loss = b_xent(logits, lbl)

    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))

embeds, _ = model.embed(features, sp_nor_adjs if sparse else nor_adjs, sparse, None)
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

tot = torch.zeros(1)
tot = tot.cuda()
tot_mac = 0
accs = []
mac_f1 = []

for _ in range(5):
    bad_counter = 0
    best = 10000
    loss_values = []
    best_epoch = 0
    patience = 20
    
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    if torch.cuda.is_available():
       log.cuda()

    for epoch in range(10000):
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
        
        if bad_counter == patience:
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
print(embeds.size())
embeddings = embeds.to(torch.device("cpu")).squeeze(0).detach().numpy()   
print(embeddings.shape)
print(labels.size())
labels_np = labels.squeeze(0)
print(labels_np.size())
#k=labels_np.max().item() + 1
#print(k)
labels_np = labels_np.to(torch.device("cpu")).numpy()
clustering.my_Kmeans(embeddings, labels_np, k=4, time=10, return_NMI=False)