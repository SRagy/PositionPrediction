from functools import partial
from typing import Optional
from warnings import warn
import torch
from torch import Tensor, nn, relu, tanh, tensor, uint8
from sbi.utils import posterior_nn

from nflows import distributions as distributions_
from nflows import flows, transforms
from nflows.nn import nets
from nflows.transforms.splines import rational_quadratic
from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

class sbi_nn(torch.nn.Module):
    def __init__(self, sample, type='nsf'):
        # type one of 'maf', 'mdn', 'nsf', 'made'
        super().__init__()
        nn_builder = posterior_nn(model = type)
        #theta_sample = gen_model.prior.sample((2,))
        theta_sample = sample
        self.dummy_x = torch.zeros(2,1)
        self.nn = nn_builder(theta_sample, self.dummy_x)
        self.params = self.nn.parameters()

    def log_prob(self, theta):
        batch_dim = theta.size(0)
        dummy_x = torch.zeros(batch_dim,1)
        return self.nn.log_prob(theta, context=dummy_x)

    def sample(self, num_samples):
        return self.nn.sample(num_samples, context = torch.tensor([[0.]])).reshape(num_samples,-1)
