"""
HiFi-GAN Vocoder (Voice)
Neural vocoder for high-fidelity audio synthesis from mel-spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=((kernel_size - 1) * d) // 2)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=80, upsample_rates=(8, 8, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4)):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        
        self.conv_pre = nn.Conv1d(in_channels, 512, 7, 1, padding=3)
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(512 // (2**i), 512 // (2**(i + 1)), k, u, padding=(k - u) // 2))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 512 // (2**(i + 1))
            self.resblocks.append(ResBlock(ch, 3))
            
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x = self.resblocks[i](x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class HiFiGANVocoder(nn.Module):
    """
    Indigenous High-Fidelity Vocoder
    Converts Mel-spectrograms to raw audio waveforms
    """
    def __init__(self, in_channels=80):
        super().__init__()
        self.generator = Generator(in_channels=in_channels)

    @torch.no_grad()
    def forward(self, mel):
        """
        Inference: Mel -> Audio
        """
        self.eval()
        # mel shape: [batch, mel_channels, seq_len]
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0).transpose(1, 2)
        elif len(mel.shape) == 3 and mel.shape[1] != 80:
             mel = mel.transpose(1, 2)
             
        audio = self.generator(mel)
        return audio.squeeze()


if __name__ == "__main__":
    vocoder = HiFiGANVocoder()
    print(f"✅ HiFi-GAN Vocoder initialized. Params: {sum(p.numel() for p in vocoder.parameters())/1e6:.1f}M")
    
    dummy_mel = torch.randn(1, 80, 100)
    audio = vocoder(dummy_mel)
    print(f"✅ Generated Audio shape: {audio.shape}")
