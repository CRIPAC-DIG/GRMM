import torch
import math
import torch.nn.functional as F
import torch.nn as nn
# https://github.com/Georgetown-IR-Lab/cedr
class PACRRConvMax2dModule(nn.Module):
    def __init__(self, shape, n_filters, k, channels):
        super().__init__()
        self.shape = shape
        if shape != 1:
            self.pad = nn.ConstantPad2d((0, shape-1, 0, shape-1), 0)
        else:
            self.pad = None
        self.conv = nn.Conv2d(channels, n_filters, shape)
        self.activation = nn.ReLU()
        self.k = k
        self.shape = shape
        self.channels = channels

    def forward(self, simmat):
        BATCH, CHANNEL,  QLEN, DLEN = simmat.shape
        if self.pad:
            simmat = self.pad(simmat)
        conv = self.activation(self.conv(simmat))#batch, channels, q_len, d_len
        top_filters, _ = conv.max(dim=1) #batch, q_len, d_len
        top_toks, _ = top_filters.topk(self.k, dim=2) # batch, q_len, k
        result = top_toks.reshape(BATCH, QLEN, self.k)
        return result

class DRMMLogCountHistogram(nn.Module):
    def __init__(self, bins, pad_idx):
        super().__init__()
        self.bins = bins
        self.pad_idx = pad_idx

    def forward(self, simmat, dtoks, qtoks):
        # THIS IS SLOW ... Any way to make this faster? Maybe it's not worth doing on GPU?
        BATCH, QLEN, DLEN = simmat.shape
        # +1e-5 to nudge scores of 1 to above threshold
        bins = ((simmat + 1.000001) / 2. * (self.bins - 1)).int()
        # set weights of 0 for padding (in both query and doc dims)
        weights = ((dtoks != self.pad_idx).reshape(BATCH, 1, DLEN).expand(BATCH, QLEN, DLEN) * \
                  (qtoks != self.pad_idx).reshape(BATCH, QLEN, 1).expand(BATCH, QLEN, DLEN)).float()

        # no way to batch this... loses gradients here. https://discuss.pytorch.org/t/histogram-function-in-pytorch/5350
        bins, weights = bins.cpu(), weights.cpu()
        histogram = []
        for superbins, w in zip(bins, weights):
            result = []
            # for b in superbins:
            #     result.append(torch.stack([torch.bincount(q, x, self.bins) for q, x in zip(b, w)], dim=0))
            for q, x in zip(superbins, w):
                result.append(torch.bincount(q, x, self.bins))
            result = torch.stack(result, dim=0)
            histogram.append(result)
        histogram = torch.stack(histogram, dim=0)

        # back to GPU
        histogram = histogram.to(simmat.device)
        return (histogram.float() + 1e-5).log()


class KNRMRbfKernelBank(nn.Module):
    def __init__(self, mus=None, sigmas=None, dim=1, requires_grad=True):
        super().__init__()
        self.dim = dim
        kernels = [KNRMRbfKernel(m, s, requires_grad=requires_grad) for m, s in zip(mus, sigmas)]
        self.kernels = nn.ModuleList(kernels)

    def count(self):
        return len(self.kernels)

    def forward(self, data):
        return torch.stack([k(data) for k in self.kernels], dim=self.dim)


class KNRMRbfKernel(nn.Module):
    def __init__(self, initial_mu, initial_sigma, requires_grad=True):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(initial_mu), requires_grad=requires_grad)
        self.sigma = nn.Parameter(torch.tensor(initial_sigma), requires_grad=requires_grad)

    def forward(self, data):
        adj = data - self.mu
        return torch.exp(-0.5 * adj * adj / self.sigma / self.sigma)


