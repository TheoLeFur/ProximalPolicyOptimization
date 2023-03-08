import attr
import torch
import torch.nn as nn
import numpy as np
from infrastructure.replay_buffer import ReplayBuffer
from agents.base_agent import BaseAgent
from policies.MLPPolicy import PPOPolicy
from critics.PPOCritic import PPOCritic


class PPOAgent:

    def __init__(self, hparams: dict, gamma: float = 0.99, gae_lambda: float = 0.95, batch_size: int = 64, n_epochs: int = 5, N: int = 20) -> None:


        self.hparams = hparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.N = N
        self.action_dim = self.hparams["action_dim"]
        self.observation_dim = self.hparams["observation_dim"]
        self.n_layers = self.hparams["n_layers"]
        self.size = self.hparams["size"]
        self.device = self.hparams["device"]
        self.learning_rate = self.hparams["learning_rate"]
        self.discrete = self.hparams["discrete"]
        self.eps_clip = self.hparams["eps_clip"]


        self.actor = PPOPolicy(
            action_dim=self.action_dim,
            ob_dim=self.observation_dim,
            n_layers=self.n_layers,
            size=self.size,
            device=self.device,
            learning_rate=self.learning_rate,
            discrete=self.discrete,
            eps_clip=self.eps_clip
        )

        self.critic = PPOCritic(
            ob_dim=self.observation_dim,
            n_layers=self.n_layers,
            size=self.size,
            device=self.device,
            learning_rate=self.learning_rate,
            discrete=self.discrete,
        )

        self.replay_buffer = ReplayBuffer(self.batch_size)

    def add_to_replay_buffer(self, state, action, probs, vals, reward, done):
        self.replay_buffer.store_memory(
            state, action, probs, vals, reward, done)

    def train(self):

        for _ in range(self.n_epochs):

            states, actions, old_probs, values, rewards, dones, batches = self.replay_buffer.generate_batches()
            advantages = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):

                    a_t += discount * \
                        (rewards[k] + self.gamma*values[k+1]) * \
                        (1-int(dones[k])) - values[k]
                    discount *= self.gamma * self.gae_lambda

                advantages[t] = a_t

            for batch in batches:

                states, actions, old_probs, values, advantages = map(lambda x: torch.tensor(
                    x[batch], dtype=torch.float32, device=self.device), [states, actions, old_probs, values, advantages])

                actor_loss = self.actor.update(
                    states, actions, advantages, old_probs)
                critic_loss = self.critic.update(
                    states, values[batch], advantages)

                total_loss = actor_loss + 0.5 * critic_loss
                print({"total_loss": total_loss,
                      "actor_loss": actor_loss, "critic_loss": critic_loss})
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.replay_buffer.clear_memory()

    def get_action(self, obs: np.ndarray):

        action, prob = self.actor.get_action(obs)
        state = torch.tensor(obs, dtype=torch.float32, device=self.device)
        value = self.critic.forward(state)

        return action.cpu().detach().numpy(), prob.cpu().detach().numpy(), value.cpu().detach().numpy()
