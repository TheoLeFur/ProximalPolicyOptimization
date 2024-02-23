import torch
import attr
import torch.nn as nn
import numpy as np
from infrastructure.pytorch_utils import build_mlp


def weights_init_uniform(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()


def weights_init_normal(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


@attr.s(eq=False, repr=False)
class RandomNetworkDistillation(nn.Module):

    ob_dim: int = attr.ib(default=None, validator=lambda i, a, x: x > 0)
    rnd_output_dim: int = attr.ib(
        default=None, validator=lambda i, a, x: x > 0)
    n_layers: int = attr.ib(default=None, validator=lambda i, a, x: x > 0)
    size: int = attr.ib(default=None, validator=lambda i, a, x: x > 0)
    rnd_learning_rate: float = attr.ib(
        default=3e-4, validator=lambda i, a, x: x > 0)
    device: torch.device = attr.ib(default=None)

    def __attrs_post_init__(self) -> None:

        super().__init__()

        self.random_function = build_mlp(
            input_size=self.ob_dim,
            output_size=self.rnd_output_dim,
            n_layers=self.n_layers,
            size=self.size,
            init_method=weights_init_uniform
        )

        self.f_hat = build_mlp(
            input_size=self.ob_dim,
            output_size=self.rnd_output_dim,
            n_layers=self.n_layers,
            size=self.size,
            init_method=weights_init_normal
        )

        self.optimizer = torch.optim.Adam(
            self.f_hat.parameters(), lr=self.rnd_learning_rate)

        self.random_function.to(self.device)
        self.f_hat.to(self.device)

    def forward(self, obs: np.ndarray):

        observation = torch.tensor(
            obs, dtype=torch.float32, device=self.device)
        prediction = self.f_hat(observation)
        target = self.random_function(observation)

        return torch.norm(prediction - target)

    def update(self, obs: np.ndarray):

        observation = torch.tensor(
            obs, dtype=torch.float32, device=self.device)
        errors = self(observation)
        loss = errors.mean()

        return loss
