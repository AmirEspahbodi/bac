import torch.nn as nn


class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, bidirectional, num_cls, dropout_rate=0.5):
    super().__init__()
    self.rnn = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             bidirectional=bidirectional,
                             batch_first=False,
                             dropout=dropout_rate if num_layers > 1 else 0) 
    fc_input_features = hidden_size * 2 if bidirectional else hidden_size
    self.fc = nn.Linear(fc_input_features, num_cls) 
    self.dropout_fc = nn.Dropout(dropout_rate)

  def forward(self, x):
    self.rnn.flatten_parameters()
    x = x.permute(1, 0, 2)
    outputs, _ = self.rnn(x) 
    out = outputs.mean(dim=0) 
    out = self.dropout_fc(out)
    y = self.fc(out) 
    return y