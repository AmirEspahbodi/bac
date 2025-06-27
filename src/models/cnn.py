import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CNNModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        filter_sizes: List[int],
        num_filters_per_size: List[int],
        num_classes: int,
        dropout_rate: float,
        hidden_dim_fc1: int,
        hidden_dim_fc2: int
    ):
        super(CNNModel, self).__init__()

        self.conv_blocks = nn.ModuleList()
        for f_size, n_filters in zip(filter_sizes, num_filters_per_size):
            conv_layer = nn.Conv1d(
                in_channels=embed_dim, out_channels=n_filters, kernel_size=f_size
            )
            bn_layer = nn.BatchNorm1d(num_features=n_filters)
            self.conv_blocks.append(nn.Sequential(conv_layer, bn_layer))

        total_filters = sum(num_filters_per_size)

        self.dropout_conv = nn.Dropout(p=dropout_rate)
        
        # FC Layer 1: Takes concatenated conv features and outputs size hidden_dim_fc1 (256)
        self.fc1 = nn.Linear(total_filters, hidden_dim_fc1)
        self.bn_fc1 = nn.BatchNorm1d(num_features=hidden_dim_fc1)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        
        # CHANGE: FC Layer 2 is added, taking fc1's output and mapping it to hidden_dim_fc2 (128)
        self.fc2 = nn.Linear(hidden_dim_fc1, hidden_dim_fc2)
        self.bn_fc2 = nn.BatchNorm1d(num_features=hidden_dim_fc2)
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)

        # CHANGE: An output layer is added to map from fc2's output to the number of classes
        self.fc_out = nn.Linear(hidden_dim_fc2, num_classes)


    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:

        # Permute to match Conv1d expectation: (batch_size, embed_dim, seq_length)
        x_permuted = x_ids.permute(0, 2, 1)

        conv_outputs = []
        for block in self.conv_blocks:
            conv_output = block(x_permuted)
            conv_output = F.relu(conv_output)
            
            pooled_output = F.max_pool1d(
                conv_output, kernel_size=conv_output.size(2)
            ).squeeze(2)
            conv_outputs.append(pooled_output)

        # Concatenate features from all filter sizes
        x_concatenated = torch.cat(conv_outputs, dim=1)
        x_dropped_out_conv = self.dropout_conv(x_concatenated)
        
        # Fully Connected Layer 1 (Output size: 256)
        x_fc1 = self.fc1(x_dropped_out_conv)
        x_bn_fc1 = self.bn_fc1(x_fc1)
        x_activated_fc1 = F.relu(x_bn_fc1)
        x_dropped_out_fc1 = self.dropout_fc1(x_activated_fc1)
        
        # CHANGE: Forward pass through the new FC Layer 2 (Output size: 128)
        x_fc2 = self.fc2(x_dropped_out_fc1)
        x_bn_fc2 = self.bn_fc2(x_fc2)
        x_activated_fc2 = F.relu(x_bn_fc2)
        x_dropped_out_fc2 = self.dropout_fc2(x_activated_fc2)
        
        # CHANGE: Pass through the final output layer to get logits
        logits = self.fc_out(x_dropped_out_fc2)
        
        return logits