import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that uses RNN (Recurrent Neural Network) layer.

    Semantics:
        output, h_n = RNN(x, h_0)
        Applies a multi-layer Elman RNN to an input sequence.
        Input shape: (seq_length, batch_size, input_size)
        Output shape: (seq_length, batch_size, hidden_size)
    """

    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=False)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (seq_length, batch_size, input_size)
            h_0: Initial hidden state of shape (1, batch_size, hidden_size)
        
        Returns:
            output: Tensor of shape (seq_length, batch_size, hidden_size)
        """
        output, h_n = self.rnn(x, h_0)
        return output


# ======== Example input configuration ========

batch_size = 1
seq_length = 8
input_size = 10
hidden_size = 20

def get_inputs():
    x = torch.randn(seq_length, batch_size, input_size)
    h_0 = torch.randn(1, batch_size, hidden_size)
    return [x, h_0]

def get_init_inputs():
    return []