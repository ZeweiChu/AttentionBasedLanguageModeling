import numpy as np
from collections import Counter
import torch
import code

# some global variable.
vocab = dict()
vocab['@'] = 0
n_vocab = 0

#####  char language modeling helper functions
# pad the seq with 0    
def make_mask(data):

    length = [len(seq) for seq in data]
    max_len = max(length)
    masked_data = []
    
    for i, seq in enumerate(data):
        seq_pad = np.zeros(max_len, dtype=np.int32)
        seq_pad[0:length[i]] = seq
        masked_data.append(seq_pad)
        
    return masked_data
    
# load the data and transform the char to integer.    
def load_data_onechar(filename, nocaps=True):

    global vocab, n_vocab
    dataset = []
    word_cnt = 0
    lines = open(filename,"rb").readlines()

    for line in lines:
        raw = '{'+line.decode('utf-8', errors='replace').strip().lower()+'}'
        if nocaps: 
            raw = replace_caps(raw)
       
        chars = [char for char in raw]
        word_cnt += len(chars)
        idx = np.ndarray((len(chars),), dtype=np.int32)
        for i, char in enumerate(chars):
            if char not in vocab:
                vocab[char] = len(vocab)
            idx[i] = vocab[char]
        dataset.append(idx)
            
    n_vocab = len(vocab) 
    dataset.sort(key=len)
    
    return dataset, word_cnt

# given integer and return the corresponding the char
def to_word(idx):
    global vocab
    return list(vocab.keys())[list(vocab.values()).index(idx)]

# given the corresponding the char and return the index 
def to_index(word):
    global vocab
    return vocab.get(word.lower(), vocab.get('{', 0))

# given integers and return the corresponding the chars
def to_string(idxs):
    string = [to_word(idx) for idx in idxs]
    return ''.join(string) 

# given the corresponding the chars and return the indexes. 
def to_idxs(words):
    idxs = [to_index(word) for word in words] 
    return idxs 


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