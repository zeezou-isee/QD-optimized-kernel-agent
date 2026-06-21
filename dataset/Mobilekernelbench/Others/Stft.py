import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, window=None, onesided=True, center=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.onesided = onesided
        self.center = center
        
        if window is not None:
            self.register_buffer("window", window)
        else:
            self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x):
        """
        x: [N, L] or [L]
        Returns: STFT result in MNN format: [N, F, T, 2]
        (real+imag split)
        """

        # PyTorch STFT
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=False,
            onesided=self.onesided
        )

        # PyTorch outputs: [*, freq, frames, 2]
        return stft

n_fft = 256
hop_length = 64
win_length = 200

# ======== Example input configuration ========
def get_inputs() -> list:
    x = torch.randn(4, 16000) * 0.1
    return [x]

def get_init_inputs():
    return [n_fft, hop_length, win_length]
