import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, bidirectional, num_cls
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.fc = nn.LazyLinear(
            num_cls
        )  # automatically infers the input size the first time it sees data

    def forward(self, x):
        outputs, _ = self.rnn(
            x
        )  # outputs:hidden states for each time step. _:final hidden state (ignored)
        out = outputs.mean(dim=0)
        y = self.fc(out)

        return y

class LSTMModelO(nn.Module):
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