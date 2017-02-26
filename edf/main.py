import utils
import numpy as np
import edf
from time import time
import pickle
import os
import torch
from torch.autograd import Variable
from model2 import RNNModel
import math

train_data, trcnt = utils.load_data_onechar('data/ptb.train.txt', False)
valid_data, vacnt = utils.load_data_onechar('data/ptb.valid.txt', False)
test_data, tecnt = utils.load_data_onechar('data/ptb.test.txt', False)


def CalPerp(output, target, masks):
    prob = output.gather(1, target).data#.numpy()
    masks = masks.data#.numpy()
    #code.interact(local=locals())

    return -torch.sum(prob[(prob*masks) < 0])

def Eval(data, cnt, model):
    
    perp = 0.
    avg_loss = 0.
    test_batches = range(0, len(data)- batch, batch)
    test_minbatches = [data[idx:idx+batch] for idx in test_batches]
    
    for minbatch in test_minbatches:
        
        x_padded = utils.make_mask(minbatch)
        
        x_padded = repackage_variable(x_padded)
        x_padded = torch.cat(x_padded, 1)
        T = x_padded.size(0)
        B = x_padded.size(1)
        inp = x_padded[:T-1, :].long()
        target = x_padded[1:, :].long().view(-1, 1)
        
        mask = (inp != 0).float().view(-1, 1)
        
        hidden = model.init_hidden(batch)
        model.zero_grad()
        output, hidden = model(inp, hidden)
        output = output.view(-1, n_vocab)
        
        loss = output.gather(1, target) * mask
        loss = -torch.sum(loss) / torch.sum(mask)
        
        avg_loss += loss
        perp += CalPerp(output, target, mask)
        #print("finish iteration")
           
    perp = np.exp(perp/cnt)
    avg_loss /= len(test_batches)
    return perp, avg_loss

def Predict(max_step, prefix, model):
    T = max_step       
    prediction = []

    for t in range(T):
   
        if t < len(prefix):
            pred = prefix[t]
            prediction.append(pred)              
        else:
            prediction.append(pred)
        
        
        hidden = model.init_hidden(1)
        inp = Variable(torch.LongTensor([[pred]]))
        #print(inp)
        output, hidden = model(inp, hidden)
        output = output.view(n_vocab)
        #print(output)
        pred = output.data.max(0)[1][0]
        #print(pred)
        
    
    idx = [pred for pred in prediction]
    stop_idx = utils.to_index('}')
    
    if stop_idx in idx:
        return idx[0:idx.index(stop_idx)+1]
    else:
        return idx  


hidden_dim = 200
n_vocab = utils.n_vocab
batch = 50
eta = 0.5
decay = 0.9


def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

def repackage_variable(v):
    return [Variable(torch.from_numpy(h)).unsqueeze(1) for h in v]

batches = range(0, len(train_data) - batch, batch)
minbatches = [train_data[idx:idx+batch] for idx in batches]

epoch = 30

model = RNNModel("LSTMCell", n_vocab, hidden_dim, hidden_dim, 1)
#model = torch.load("model_att_torch.pkl")
crit = torch.nn.CrossEntropyLoss()

prefix = 'the agreements bring'  
generation = Predict(400, utils.to_idxs(prefix), model)
print("generated sentence ")
print(utils.to_string(generation))

perp, loss = Eval(valid_data, vacnt, model)
#print(type(perp))
#print(type(loss))
print("Initial: Perplexity: "+str(perp) + "Avg loss = " + str(loss) )    
best_loss = loss



for ep in range(epoch):

    perm = np.random.permutation(len(minbatches)).tolist() 
    stime=time()
    
    for k in range(len(minbatches)):
        
        minbatch = minbatches[perm[k]]
        x_padded = utils.make_mask(minbatch)
        x_padded = repackage_variable(x_padded)
        x_padded = torch.cat(x_padded, 1)
        T = x_padded.size(0)
        B = x_padded.size(1)
        inp = x_padded[:T-1, :].long()
        target = x_padded[1:, :].long().view(-1, 1)
        
        mask = (inp != 0).float().view(-1, 1)
        
        hidden = model.init_hidden(batch)
        model.zero_grad()
        #print(inp.size())
        output, hidden = model(inp, hidden)
        output = output.view(-1, n_vocab)
        
        loss = output.gather(1, target) * mask
        loss = -torch.sum(loss) / torch.sum(mask)
        loss.backward()
        
        #clipped_lr = lr * clip_gradient(model, args.clip)
        clipped_lr = eta * clip_gradient(model, 0.5)
        for p in model.parameters():
            p.data.add_(-clipped_lr, p.grad.data)
        
        
        if k % 200 == 0:
            #print(loss.data)
            #output = output.max(1)[1]
            acc = torch.sum((output.max(1)[1] == target).float() * mask) / torch.sum(mask)
            print("Batch accuracy: " + str(acc.data))
      
    
    prefix = 'the agreements bring'  
    generation = Predict(400, utils.to_idxs(prefix), model)
    print("generated sentence ")
    print(utils.to_string(generation))

    perp, loss = Eval(valid_data, vacnt, model)
    print("Perplexity: "+str(perp) + "Avg loss = " + str(loss) )
    
    if loss < best_loss:
        torch.save(model, "model_att_torch.pkl")
        best_loss = loss
    else:
        eta *= decay
        model = torch.load("model_att_torch.pkl")
        



        