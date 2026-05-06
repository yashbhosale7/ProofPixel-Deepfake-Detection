import torch
import torch.nn as nn

class VideoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, T, D]
        _, (hn, _) = self.lstm(x)
        out = hn[-1]            # last layer hidden state
        out = self.fc(out)
        return out
