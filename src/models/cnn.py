import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        filter_sizes,
        num_filters_per_size,
        num_classes,
        dropout_rate,
        hidden_dim_fc,
        l2_lambda=0.001,
    ):
        super(CNNModel, self).__init__()

        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=n_filters, kernel_size=f_size
                )
                for f_size, n_filters in zip(filter_sizes, num_filters_per_size)
            ]
        )

        total_filters = sum(num_filters_per_size)
        self.dropout_conv = nn.Dropout(p=dropout_rate) 
        self.fc1 = nn.Linear(total_filters, hidden_dim_fc)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate) 
        self.fc2 = nn.Linear(hidden_dim_fc, num_classes)
        self.l2_lambda = l2_lambda 

    def forward(self, x):
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv_layer in self.conv1d_list:
            conv_output = conv_layer(x)
            conv_output = F.relu(conv_output)
            conv_output = F.max_pool1d(
                conv_output, kernel_size=conv_output.size(2)
            ).squeeze(2)
            conv_outputs.append(conv_output)

        x_concatenated = torch.cat(conv_outputs, dim=1)
        x_dropped_out_conv = self.dropout_conv(x_concatenated) 
        x_fc1 = F.relu(self.fc1(x_dropped_out_conv))
        x_dropped_out_fc1 = self.dropout_fc1(x_fc1) 
        logits = self.fc2(x_dropped_out_fc1)
        
        return logits