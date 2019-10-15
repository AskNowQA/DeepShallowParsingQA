import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    def __init__(self, vocab_size, emb_size, input_size, hidden_size, output_size, dropout_ratio, emb_idx,
                 bidirectional=False):
        super(LSTMPolicy, self).__init__()
        self.output_size = output_size
        self.emb_idx = emb_idx

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0, sparse=False)
        self.emb.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout_ratio, bidirectional=bidirectional)
        self.hidden2action = nn.Linear(hidden_size + (bidirectional * hidden_size), output_size)
        self.activation = nn.Softmax(dim=2)
        self.hidden = None

    def init(self):
        self.hidden = None

    def forward(self, input, hidden=None):
        input = torch.cat((input[:self.emb_idx].float().reshape(-1),
                           self.emb(input[self.emb_idx:]).reshape(-1)))
        if hidden is None:
            hidden = self.hidden
        if hidden is None:
            lstm_out, hidden = self.lstm(input.reshape(1, 1, -1))
        else:
            lstm_out, hidden = self.lstm(input.reshape(1, 1, -1), hidden)
        action_space = self.hidden2action(lstm_out)
        action_score = self.activation(action_space)

        self.hidden = hidden
        return action_score.reshape(-1)
