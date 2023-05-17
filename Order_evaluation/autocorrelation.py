import torch
import torch.nn as nn

class autocorrelation_block(nn.Module):
    def __init__(self, c: float, heads: int, output: int, trainable: bool, d_model: int):
        super(autocorrelation_block, self).__init__()
        self.fft = torch.fft.rfft
        self.inv_fft = torch.fft.irfft
        self.heads = heads
        self.out_ = output
        self.c = c
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=0)
        self.trainable = trainable

    def forward(self, q, k, v):
        # Get B, L, d values
        B, L, d = q.shape
        # Resize. Fill with zeros if S < L. Else, remove items so that the shapes match.
        if k.shape[1] < L:
            k = torch.cat([k, torch.zeros(size=(B, L - k.shape[1], d), device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(size=(B, L - v.shape[1], d), device=v.device)], dim=1)
        elif k.shape[1] > L:
            k = k[:, :L, :]
            v = v[:, :L, :]

        # Reshape to same shape
        dh = int(self.d_model / self.heads)
        if dh * self.heads * L * B < B * L * d:
            dh = dh + 1
        # We always use multihead attention so
        k = k.view([B, L, self.heads, dh])
        v = v.view([B, L, self.heads, dh])
        q = q.view([B, L, self.heads, dh])

        # FFT transform
        q, k = self.fft(q, dim=1), self.fft(k, dim=1)
        # Complex conjugate inside inverse fft provides us with autocorrelation
        corr = q * torch.conj(k)
        corr = self.inv_fft(corr, dim=1)
        # Take the mean of results to simplify
        corr = torch.mean(corr, dim=(0, 2, 3))
        # Get indices of max elements
        k = torch.floor(self.c * torch.log(torch.tensor(L, dtype=torch.int64)).float())
        k = int(k)
        weights = torch.topk(corr, k, dim=0)
        # Get max weights from corr (e.g. the values that are most closely correlated)
        indices = weights.indices.tolist()
        # Softmax to balance the weights. Sum of softmax values = 1
        weights = self.softmax(weights.values)
        # If trainable, use roll. Otherwise, get the max values
        if self.trainable:
            for i in range(int(torch.floor(self.c * torch.log(torch.tensor(L, dtype=torch.int64)).float()))):
                val = torch.roll(v, indices[i], dims=1)
                weight = weights[i]
                R = [weight * val]
        else:
            idx = []
            for _ in range(d):
                idx.append(torch.arange(0, L, 1, device=v.device))
            idx = torch.stack(idx).reshape(B, L, self.heads, int(d / self.heads))
            v = v.repeat(1, 2, 1, 1).reshape(B, 2 * L, self.heads, int(d / self.heads))
            R = [weights[i] * torch.take(v, (indices[i] + idx)) for i in
                 range(int(torch.floor(self.c * torch.log(torch.tensor(L, dtype=torch.int64)).float())))]
        R = torch.sum(torch.stack(R, dim=0), dim=0)
        return R.view([B, L, self.d_model])

