import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


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
    x = x.permute(0, 2, 1)
    outputs, _ = self.rnn(x) # outputs: (seq_len, batch, num_directions * hidden_size)
    # Aggregate RNN outputs. Using mean pooling over time steps.
    out = outputs.mean(dim=0) # Shape: (batch, num_directions * hidden_size)
    out = self.dropout_fc(out)
    y = self.fc(out) # Shape: (batch, num_cls)
    return y