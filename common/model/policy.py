import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_ratio):
        super(Policy, self).__init__()

        self.activation0 = nn.GLU()
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        self.activation1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.layer2 = nn.Linear(int(hidden_size), output_size, bias=False)
        self.activation2 = nn.Softmax()

    def forward(self, input):
        output_layer1 = self.activation1(self.layer1(input))
        output_layer1 = self.dropout(output_layer1)
        output_layer2 = self.activation2(self.layer2(output_layer1))
        return output_layer2
