import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class LSTMConfig:
    input_size: int = 768 
    hidden_size: int = 512
    num_layers: int = 2
    num_classes: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    use_attention: bool = False
    attention_heads: int = 8
    attention_dropout: float = 0.1
    layer_norm: bool = True
    residual_connections: bool = True
    gradient_clipping: float = 1.0
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.input_size > 0, "input_size must be positive"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert self.attention_heads > 0, "attention_heads must be positive"


class LuongAttention(nn.Module):
    """
    Luong Attention Mechanism Implementation
    Supports general, dot, and concat attention methods
    """

    def __init__(self, hidden_size: int, num_heads: int = 1, method: str = 'general'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.method = method
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.randn(hidden_size))
        elif method != 'dot':
            raise ValueError(f"Unknown attention method: {method}")
            
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, hidden_size] - typically the last hidden state
            keys: [batch_size, seq_len, hidden_size] - all hidden states
            values: [batch_size, seq_len, hidden_size] - all hidden states
            mask: [batch_size, seq_len] - padding mask
            
        Returns:
            context: [batch_size, hidden_size] - attended context vector
            attention_weights: [batch_size, seq_len] - attention weights
        """
        batch_size, seq_len, hidden_size = keys.size()
        
        if self.method == 'dot':
            # query: [batch_size, 1, hidden_size]
            scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))  # [batch_size, 1, seq_len]
            scores = scores.squeeze(1)  # [batch_size, seq_len]
            
        elif self.method == 'general':
            # Transform query and compute attention scores
            transformed_query = self.attn(query).unsqueeze(1)  # [batch_size, 1, hidden_size]
            scores = torch.bmm(transformed_query, keys.transpose(1, 2)).squeeze(1)  # [batch_size, seq_len]
            
        elif self.method == 'concat':
            # Expand query to match sequence length
            expanded_query = query.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]
            concat_input = torch.cat([expanded_query, keys], dim=2)  # [batch_size, seq_len, hidden_size*2]
            scores = torch.tanh(self.attn(concat_input))  # [batch_size, seq_len, hidden_size]
            scores = torch.sum(scores * self.v, dim=2)  # [batch_size, seq_len]
        
        # Apply scaling
        scores = scores / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [batch_size, hidden_size]
        
        return context, attention_weights


class MultiHeadLuongAttention(nn.Module):
    """Multi-head version of Luong Attention"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = keys.size()
        
        # Linear transformations and reshape for multi-head
        Q = self.q_linear(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(keys).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(values).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, 1, seq_len]
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, 1, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, self.hidden_size)
        
        # Final linear transformation
        context = self.out_linear(context)
        
        # Average attention weights across heads for visualization
        avg_attention = attention_weights.mean(dim=1).squeeze(1)  # [batch_size, seq_len]
        
        return context, avg_attention


class LSTMModel(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Attention mechanism
        if config.use_attention:
            if config.attention_heads > 1:
                self.attention = MultiHeadLuongAttention(
                    lstm_output_size, 
                    config.attention_heads,
                    config.attention_dropout
                )
            else:
                self.attention = LuongAttention(lstm_output_size, method='general')
        else:
            self.attention = None
            
        # Layer normalization
        if config.layer_norm:
            self.layer_norm1 = nn.LayerNorm(lstm_output_size)
            self.layer_norm2 = nn.LayerNorm(lstm_output_size)
        else:
            self.layer_norm1 = None
            self.layer_norm2 = None
            
        # Dropout layers
        self.dropout = nn.Dropout(config.dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Output layers
        classifier_input_size = lstm_output_size
        if config.use_attention:
            classifier_input_size = lstm_output_size  # Context vector size
            
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(classifier_input_size // 2, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
         
    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        self.lstm.flatten_parameters()
        # inputs = inputs.permute(1, 0, 2)
        
        # Input projection
        # projected_inputs = self.input_projection(inputs)
        # projected_inputs = self.dropout(projected_inputs)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(inputs)
        
        # Apply layer normalization
        if self.layer_norm1 is not None:
            lstm_out = self.layer_norm1(lstm_out)

        # if self.layer_norm2 is not None:
        #     hidden = self.layer_norm2(hidden)
            
        # Store hidden states for output
        hidden_states = lstm_out
        
        # Attention mechanism
        attention_weights = None
        if self.attention is not None:
            # Create padding mask if lengths are provided
            mask = None
            if lengths is not None:
                mask = self.create_padding_mask(inputs, lengths)
            
            # Use the last hidden state as query
            if self.config.bidirectional:
                # For bidirectional LSTM, concatenate forward and backward final states
                query = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_size*2]
            else:
                query = hidden[-1]  # [batch_size, hidden_size]
                
            # Apply attention
            context, attention_weights = self.attention(query, lstm_out, lstm_out, mask)
            
            # Apply layer normalization and residual connection
            if self.config.residual_connections:
                context = context + query

            if self.layer_norm2 is not None:
                context = self.layer_norm2(context)
                
            final_representation = self.attention_dropout(context)
        else:
            # Use the final hidden state without attention
            if self.config.bidirectional:
                final_representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                final_representation = hidden[-1, :, :]
                
        # Classification
        logits = self.classifier(final_representation)
        
        return logits

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)
                
    def create_padding_mask(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention mechanism"""
        batch_size, max_len = inputs.size(0), inputs.size(1)
        mask = torch.arange(max_len, device=inputs.device).expand(batch_size, max_len)
        mask = mask < lengths.unsqueeze(1)
        return mask
       
       
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights"""
        return getattr(self, '_last_attention_weights', None)



def get_model_info(model: LSTMModel) -> Dict[str, Any]:
    """Get model information and statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
        'config': model.config,
        'is_bidirectional': model.config.bidirectional,
        'uses_attention': model.config.use_attention,
        'attention_heads': model.config.attention_heads if model.config.use_attention else 0
    }


