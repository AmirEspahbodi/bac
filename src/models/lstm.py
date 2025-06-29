import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LSTMModelO(nn.Module):
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
    x = x.permute(1, 0 , 2)
    x = x.contiguous()
    outputs, _ = self.rnn(x) # outputs: (seq_len, batch, num_directions * hidden_size)
    # Aggregate RNN outputs. Using mean pooling over time steps.
    out = outputs.mean(dim=0) # Shape: (batch, num_directions * hidden_size)
    out = self.dropout_fc(out)
    y = self.fc(out) # Shape: (batch, num_cls)
    return y

class LSTMModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        _packed_output, (hidden, _cell) = self.lstm(text_embeddings)

        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        output = self.fc(hidden)
        return output