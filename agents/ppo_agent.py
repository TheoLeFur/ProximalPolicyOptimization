import attr
import torch
import torch.nn as nn
import numpy as np
from infrastructure.replay_buffer import ReplayBuffer
from agents.base_agent import BaseAgent
from policies.MLPPolicy import PPOPolicy
from critics.PPOCritic import PPOCritic


@attr.s(eq=False, repr=False)
class PPOAgent(BaseAgent):

    hparams = attr.ib()
    gamma: float = attr.ib(default=0.99, validator=lambda i, a, x: x > 0)
    gae_lambda: float = attr.ib(default=0.99, validator=lambda i, a, x: x > 0)
    batch_size: int = attr.ib(default=16)
    n_epochs: int = attr.ib(default=7)
    N: int = attr.ib(default=50)
    entropy_reg: float = attr.ib(default=0.01)

    def __attrs_post_init__(self):


        self.action_dim = self.hparams["action_dim"]
        self.observation_dim = self.hparams["observation_dim"]
        self.n_layers = self.hparams["n_layers"]
        self.size = self.hparams["size"]
        self.device = self.hparams["device"]
        self.learning_rate = self.hparams["learning_rate"]
        self.discrete = self.hparams["discrete"]
        self.eps_clip = self.hparams["eps_clip"]

        self.actor = PPOPolicy(
            self.action_dim,
            self.observation_dim,
            self.n_layers,
            self.size,
            self.device,
            self.learning_rate,
            self.discrete,
            self.eps_clip
        )

        self.critic = PPOCritic(
            self.observation_dim,
            self.n_layers,
            self.size,
            self.device,
            self.learning_rate,
            self.discrete,
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

                a_t = 0
                discount = 1

                for k in range(t, len(rewards) - 1):
                    delta_t = rewards[k] + self.gamma * \
                        values[k+1] * (1 - dones[k]) - values[k]
                    a_t += discount * (delta_t)
                advantages[t] = a_t

            

            for batch in batches:

                states, actions, old_probs, values, advantages =  map(lambda x: torch.tensor(x, dtype = torch.float32, device = self.device), [states, actions, old_probs, values, advantages])
                states, actions, old_probs, values, advantages = map(lambda x : x[batch], [states, actions, old_probs, values, advantages])
                actor_loss, entropy = self.actor.update(
                    states, actions, old_probs, advantages)
                critic_loss = self.critic.update(states, values, advantages)
                total_loss = actor_loss + critic_loss + self.entropy_reg * entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            self.replay_buffer.clear_memory()

    def get_action(self, obs: np.ndarray):

        action, prob = self.actor.get_action(obs)
        value = self.critic(torch.tensor(obs, dtype = torch.float32, device = self.device))

        return action, prob, value.cpu().detach().numpy()
