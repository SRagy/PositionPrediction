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


class UnconditionalFlow(torch.nn.Module):
    def __init__(self, sample: Tensor, type: str = 'nsf', device = torch.device('cpu')):
        """A normalising flow with architecture import from the sbi
        module.

        Args:
            sample (Tensor): n x m sample from the data, where n is training batch size.
            type (str, optional): Choose flow type from 'maf', 'mdn', 'nsf', 'made'. Defaults to 'nsf'.
            device (device, optional): Choose device (cpu, gpu etc.)
        """
        # type one of 'maf', 'mdn', 'nsf', 'made'
        super().__init__()
        nn_builder = posterior_nn(model = type)
        theta_sample = sample.to(device)
        # Make flow unconditional by setting a fixed conditioning variable.
        self.dummy_x = torch.zeros(2,1).to(device)
        self.nn = nn_builder(theta_sample, self.dummy_x)
        self.params = self.nn.parameters()
        self.device = device

    def log_prob(self, theta: Tensor):
        """Returns log probability density of sample theta.

        Args:
            theta (Tensor): n x m batch of samples.

        Returns:
            Tensor: n x 1 batch of log probabilities.
        """
        batch_dim = theta.size(0)
        dummy_x = torch.zeros(batch_dim,1).to(self.device)
        return self.nn.log_prob(theta, context=dummy_x)

    def sample(self, num_samples):
        context = torch.tensor([[0.]]).to(self.device)
        return self.nn.sample(num_samples, context = context).reshape(num_samples,-1)
