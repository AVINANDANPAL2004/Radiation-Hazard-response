"""Q-networks: baseline MLP, CNN-DRL, CNN-LSTM-DRL (also used by the LLM-guided agent)."""
import os
import torch
import torch.nn as nn
from env import N, VEC_DIM, N_ACTIONS


class FrameEncoder(nn.Module):
    """CNN over the 4-channel facility image + MLP over the sensor vector."""
    def __init__(self, out=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(32 * (N // 2) ** 2, 64), nn.ReLU())
        self.vec = nn.Sequential(nn.Linear(VEC_DIM, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(64 + 128, out), nn.ReLU())
        # gated image pathway: starts at 0 (agent behaves like the sensor-only baseline) and
        # the network learns how much to trust the image features
        self.gate = nn.Parameter(torch.zeros(1))
        self.use_ln = os.environ.get("ENC_LN", "0") == "1"
        self.ln = nn.LayerNorm(64)

    def forward(self, img, vec):
        c = self.cnn(img)
        if self.use_ln:
            c = self.ln(c)
        return self.fuse(torch.cat([self.gate * c, self.vec(vec)], dim=-1))


class QNet(nn.Module):
    """kind: 'baseline' (MLP on sensor vector only), 'cnn' (single frame),
    'cnn_lstm' (window of K frames -> LSTM)."""
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "baseline":
            self.body = nn.Sequential(nn.Linear(VEC_DIM, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU())
        else:
            self.enc = FrameEncoder(128)
            if kind == "cnn_lstm":
                self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, N_ACTIONS))

    def forward(self, img, vec):
        """img: (B,K,4,N,N), vec: (B,K,V)."""
        B, K = vec.shape[:2]
        if self.kind == "baseline":
            h = self.body(vec[:, -1])
        elif self.kind == "cnn":
            h = self.enc(img[:, -1], vec[:, -1])
        else:
            z = self.enc(img.reshape(B * K, 4, N, N), vec.reshape(B * K, -1)).reshape(B, K, -1)
            h = self.lstm(z)[0][:, -1]
        return self.head(h)
