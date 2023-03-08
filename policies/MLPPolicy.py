import torch
import torch.nn as nn
import numpy as np
import itertools
import torch.distributions as distributions
import attr
from typing import Union
from collections import OrderedDict
from infrastructure.pytorch_utils import build_mlp


@attr.s(eq=False, repr=False)
class PPOPolicy(nn.Module):

    action_dim: int = attr.ib(default=None),
    ob_dim: int = attr.ib(default=None),
    n_layers: int = attr.ib(default=2, validator=lambda i, a, x: x > 0),
    size: int = attr.ib(default=64, validator=lambda i, a, x: x > 0)
    device: torch.device = attr.ib(default=None),
    learning_rate: float = attr.ib(
        default=3e-4, validator=lambda i, a, x: x > 0),
    discrete: bool = attr.ib(default=True),
    eps_clip: float = attr.ib(default=None, validator=lambda i, a, x: x > 0)

    def __attrs_post_init__(self) -> None:

        super().__init__()

        if self.discrete:

            self.logits_na = build_mlp(
                input_size=self.ob_dim,
                output_size=self.action_dim,
                n_layers=self.n_layers,
                size=self.size,
                output_activation='softmax')

            self.logits_na.to(self.device)
            self.mean_net = None
            self.log_std = None
            self.optimizer = torch.optim.Adam(
                self.logits_na.parameters(), lr=self.learning_rate)

        else:
            self.logits_na = None
            self.mean_net = self.build_mlp(
                input_size=self.ob_dim,
                output_size=self.action_dim,
                n_layers=self.n_layers,
                size=self.size
            )
            self.mean_net.to(self.device)
            self.log_std = nn.Parameter(torch.zeros(
                self.action_dim, dtype=torch.float32, device=self.device))
            self.optimizer = torch.optim.Adam(itertools.chain(
                [self.log_std], self.mean_net.parameters()), lr=self.learning_rate)

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        if len(observation) > 1:
            observation = observation[None]
        with torch.no_grad():
            distribution = self.forward(torch.tensor(
                [observation]).to(self.device))
            action = distribution.sample()
            logprobs = distribution.log_prob(action)

        return torch.squeeze(action), torch.squeeze(logprobs)

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.Tensor) -> distributions.Distribution:
        if self.discrete:
            logits = self.logits_na(observation).squeeze()
            return torch.distributions.Categorical(logits=logits)

        else:
            batch_mean = self.mean_net(observation)
            covariance = torch.exp(self.log_std)
            return torch.distributions.Normal(batch_mean, covariance)

    def update(self, observations: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor, old_log_probs: torch.Tensor):

        distribution = self(observations)
        new_log_probs = distribution.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = advantages * ratio
        surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                            1 + self.eps_clip) * advantages

        actor_loss = - torch.min(surr1, surr2).mean()
        return actor_loss
