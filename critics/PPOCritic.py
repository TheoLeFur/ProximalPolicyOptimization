import torch
import torch.nn as nn
import numpy as np
from critics.BaseCritic import BaseCritic
from infrastructure.pytorch_utils import build_mlp


class PPOCritic(nn.Module):

    def __init__(
            self,
            ob_dim: np.ndarray, 
            n_layers: int = 2, 
            size: int = 64, 
            device: torch.device = None, 
            learning_rate: float = 3e-4, 
            discrete: bool = True) -> None:

        super().__init__()

        self.ob_dim = ob_dim
        self.n_layers = n_layers 
        self.size = size 
        self.device = device 
        self.learning_rate = learning_rate
        self.discrete = discrete 

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
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        out = self.critic(obs).squeeze()
        return out.cpu().detach().numpy()

    def update(self, states: torch.Tensor, values: torch.Tensor, advantages: torch.Tensor):
        new_critic_values = self.forward(states)
        loss = self.loss_fn(advantages + values, new_critic_values)
        return loss

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)
