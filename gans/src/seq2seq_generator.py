import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=False,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        output, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell, output
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=False,
            bidirectional=False
        )
        
        self.out = nn.Linear(
            in_features=2*hid_dim,
            out_features=output_dim
        )
        
        self.w = nn.Linear(self.hid_dim, self.hid_dim)
        
        self.attn_lin = nn.Linear(self.hid_dim*2, self.hid_dim*2)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        # Compute an embedding from the input data and apply dropout to it
        embedded = self.dropout(self.embedding(input))
        
        # hidden = [layers, batch_size, hid_dim]
        # output = [seq, batch, hid_dim]
        _, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # attn_weights = [batch, seq, 1]
        attn_weights = F.softmax(
            torch.bmm(self.w(encoder_outputs.permute(1,0,2)), hidden[-1].unsqueeze(dim=2)),
            dim=1
        )
        
        #context_vector = [batch, hid_dim]
        context_vectors = torch.bmm(encoder_outputs.permute(1, 2, 0), attn_weights).squeeze(2)
        
        # output = [batch, 2hid_dim]
        output = torch.cat((hidden[-1], context_vectors), dim=1)
        output = torch.tanh(self.attn_lin(output))
    
        # prediction = [batch, output_dim]
        prediction = self.out(output.squeeze(0))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5, max_len=30):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.ones(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell, encoder_outputs = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]  # argmax
            input = (trg[t] if teacher_force else top1)
        
        return outputs
    
    
    def sample(self, src, trg, teacher_forcing_ratio=0., max_len=30):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.ones(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell, encoder_outputs = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        generated_highlight = torch.ones(max_len, batch_size).long().to(self.device)  # seq_len * batch_size
        generated_highlight[0, :] = input
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            next_token = torch.multinomial(F.softmax(output, dim=-1), 1).view(-1)
            input = (trg[t] if teacher_force else next_token)  # (batch_size,)
    
            generated_highlight[t] = input
            
        return outputs, generated_highlight
    
    
    def batch_pgloss(self, src, trg, reward):
        """
        Returns a policy gradient loss
        :param src: seq_len x batch_size
        :param trg: seq_len x batch_size
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        seq_len, batch_size = src.size()

        out = self.forward(src, trg,
                           teacher_forcing_ratio=0.).permute(1, 0, 2)  # batch * seq * vocab
        
        probs = F.softmax(out, dim=2).detach()  # batch * seq
        preds = torch.argmax(probs, dim=2)
        pred_onehot = F.one_hot(preds, self.encoder.input_dim).float()  # batch * seq * vocab
        
        target_onehot = F.one_hot(trg.permute(1, 0), self.encoder.input_dim).float()  # batch * seq * vocab
        
        out = F.log_softmax(out, dim=2)
        pred = torch.sum(out * pred_onehot, dim=-1)  # batch_size * seq_len
        # pred = torch.sum(pred, dim=-1)
        loss = -torch.mean(pred * reward)

        return loss
    

    def batch_pgloss_generated(self, gen_out, generated_highlight, reward):
  
        gen_out = gen_out.permute(1, 0, 2)  # batch * seq * vocab
        pred_onehot = F.one_hot(generated_highlight, self.encoder.input_dim).float()  # batch * seq * vocab
        out_log = F.log_softmax(gen_out, dim=2)  # batch * seq * vocab
        
        pred = torch.sum(out_log * pred_onehot, dim=-1)  # batch_size * seq
        loss = -torch.mean(pred * reward) 

        return loss