import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that uses LSTM (Long Short-Term Memory) layer.

    Semantics:
        output, (h_n, c_n) = LSTM(x, (h_0, c_0))
        Applies LSTM to an input sequence.
        Input shape: (seq_length, batch_size, input_size)
        Output shape: (seq_length, batch_size, hidden_size)
    """

    def __init__(self, input_size=16, hidden_size=32):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,
            bidirectional=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (seq_length, batch_size, input_size)
            h_0: Initial hidden state of shape (1, batch_size, hidden_size)
            c_0: Initial cell state of shape (1, batch_size, hidden_size)
        
        Returns:
            output: Tensor of shape (seq_length, batch_size, hidden_size)
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n, c_n


# ======== Example input configuration ========

batch_size = 1
seq_length = 1
input_size = 16
hidden_size = 32

def get_inputs():
    """
    Generate random input sequence and initial states for LSTM.
    """
    x = torch.randn(seq_length, batch_size, input_size)
    # h_0 = torch.randn(1, batch_size, hidden_size)
    # c_0 = torch.randn(1, batch_size, hidden_size)
    return [x]

def get_init_inputs():
    return []

