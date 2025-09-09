import scipy.stats
import torch
import torch.nn as nn

from .bound_ops import LowerBound

from ._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf

def pmf_to_quantized_cdf(pmf, precision: int = 16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf

def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
    cdf = torch.zeros(
        (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
    )
    for i, p in enumerate(pmf):
        prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
        _cdf = pmf_to_quantized_cdf(prob)
        cdf[i, : _cdf.size(0)] = _cdf
    return cdf

class generalnormalcdf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, beta, y):
        ctx.absy = torch.pow(torch.abs(y),beta)
        ctx.beta = beta
        output = (1+torch.sign(y)*torch.special.gammainc(1/beta, ctx.absy))/2
        ctx.device = y.device
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.absy
        beta = ctx.beta
        y_grad = torch.exp(-y-torch.special.gammaln(1/beta))*beta/2
        return None, grad_output*y_grad

class GGM(nn.Module):
    def __init__(
        self,
        beta: float = 1.5,
        scale_bound: float = 0.12,
        likelihood_bound: float = 1e-9
    ):
        super().__init__()

        self.scale_lower_bound = LowerBound(scale_bound)
        self.likelihood_lower_bound = LowerBound(likelihood_bound)

        self.register_buffer("beta", torch.Tensor([beta]))
    
    def _cdf(self, values):
        return generalnormalcdf.apply(self.beta, values)

    def _likelihood(self, inputs, scales, means=None):
        
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = self.scale_lower_bound(scales)
        upper = self._cdf((values+half)/scales)
        lower = self._cdf((values-half)/scales)
        likelihood = upper - lower

        return likelihood

    def forward(self, inputs, scales, means=None):
        likelihood = self._likelihood(inputs, scales, means)
        likelihood = self.likelihood_lower_bound(likelihood)
        return likelihood
    
    def _standardized_quantile(self, quantile):
        return scipy.stats.gennorm.ppf(quantile, self.beta)
    
    def get_quantized_cdf(self, scale_table):
        device = scale_table.device
        scale_table = self.scale_lower_bound(scale_table)

        tail_mass = 1e-6
        multiplier = -self._standardized_quantile(tail_mass / 2)
        pmf_center = torch.ceil(scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        samples = torch.arange(max_length, device=device).int() - pmf_center[:, None]
        samples = samples.float()
        samples_scale = scale_table.unsqueeze(1)
        samples_scale = samples_scale.float()

        upper = self._cdf((samples+0.5) / samples_scale)
        lower = self._cdf((samples-0.5) / samples_scale)
        pmf = upper - lower

        tail_mass = lower[:, :1] + (1-upper[:,-1:])

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        offset = -pmf_center
        cdf_length = pmf_length + 2
        
        return quantized_cdf, cdf_length, offset
