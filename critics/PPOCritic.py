import attr
import torch
import torch.nn as nn
import numpy as np
from critics.BaseCritic import BaseCritic
from infrastructure.pytorch_utils import build_mlp


@attr.s(eq=False, repr=False)
class PPOCritic(BaseCritic, nn.Module):

    ob_dim: int = attr.ib(validator=lambda i, a, x: x > 0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x > 0)
    size: int = attr.ib(validator=lambda i, a, x: x > 0)
    device: torch.device = attr.ib(
        default="mps" if torch.backends.mps.is_available() else "cpu")
    learning_rate: float = attr.ib(default=3e-4)
    discrete: bool = attr.ib(default=True)

    def __attrs_post_init__(self):

        super().__init__()

        self.critic = build_mlp(
            input_size=self.ob_dim,
            output_size=1,
            n_layers=self.n_layers,
            size=self.size
        )

        self.critic.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate)

    def forward(self, obs: torch.Tensor):
        return self.critic(obs).squeeze()

    def forward_np(self, obs: np.ndarray):
        obs = torch.Tensor(obs).to(self.device)
        out = self.critic(obs).squeeze()
        return out.cpu().detach().numpy()

    def update(self, states: torch.Tensor, values: torch.Tensor, advantages: torch.Tensor):
        new_critic_values = torch.squeeze(self(states))
        loss = self.loss_fn(new_critic_values, advantages + values)
        print(loss.shape, "shape")
        return loss
