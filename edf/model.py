import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnncell_type, ntoken, ninp, nhid, nlayers):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnncell = getattr(nn, rnncell_type)(ninp, nhid, bias=False)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnncell_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.logsoftmax = nn.LogSoftmax()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        h, c = hidden
        h.data = h.data.squeeze(0)
        c.data = c.data.squeeze(0)

        seq_len = input.size(0)
        batch_size = input.size(1)
        output_dim = h.size(1)

        output = [] 
        for i in range(seq_len):        
            h, c = self.rnncell(emb[i], (h, c))
            output.append(h.unsqueeze(0))
        output = torch.cat(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = self.logsoftmax(decoded)
        output = decoded.view(output.size(0), output.size(1), decoded.size(1)) 
        return output, (h, c)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        
        if self.rnn_type == 'LSTMCell':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
