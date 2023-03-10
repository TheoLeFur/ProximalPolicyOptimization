import attr
import torch
import torch.nn as nn
import numpy as np
from infrastructure.replay_buffer import ReplayBuffer
from agents.base_agent import BaseAgent
from policies.MLPPolicy import PPOPolicy
from critics.PPOCritic import PPOCritic
from exploration.random_distillation_net import RandomNetworkDistillation

@attr.s(eq = False, repr = False)
class PPOAgent():

    hparams: dict = attr.ib(default=None)
    gamma: float = attr.ib(default=0.99, validator=lambda i, a, x : x >0)
    gae_lambda: float = attr.ib(default=0.95, validator=lambda i, a, x: x > 0)
    batch_size: int = attr.ib(default = 8, validator = lambda i, a, x : x > 0)
    n_epochs: int = attr.ib(default = 5, validator = lambda i, a, x : x > 0)
    N: int = attr.ib(default = 20, validator = lambda i, a, x : x > 0)


    def __attrs_post_init__(self) -> None:

        self.action_dim = self.hparams["action_dim"]
        self.observation_dim = self.hparams["observation_dim"]
        self.n_layers = self.hparams["n_layers"]
        self.size = self.hparams["size"]
        self.device = self.hparams["device"]
        self.learning_rate = self.hparams["learning_rate"]
        self.discrete = self.hparams["discrete"]
        self.eps_clip = self.hparams["eps_clip"]
        self.rnd_output_size = self.hparams["rnd_output_size"]
        self.rnd_size = self.hparams["rnd_size"]
        self.rnd_n_layers = self.hparams["rnd_n_layers"]


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


        self.random_distillation_network = RandomNetworkDistillation(
            ob_dim = self.observation_dim,
            rnd_output_dim = self.rnd_output_size,
            n_layers = self.rnd_n_layers,
            size = self.rnd_size,
            device = self.device

        )

        self.replay_buffer = ReplayBuffer(self.batch_size)

    def add_to_replay_buffer(self, state, action, probs, vals, reward, done):
        self.replay_buffer.store_memory(
            state, action, probs, vals, reward, done)

    def train(self):

        actor_loss_buffer = []
        critic_loss_buffer = []
        total_loss_buffer = []

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

                actor_loss, entropy = self.actor.update(
                    states, actions, advantages, old_probs)
                critic_loss = self.critic.update(
                    states, values[batch], advantages)
                
                if not torch.isnan(advantages.std()):
                    advantages = (advantages - advantages.mean()) / \
                        (advantages.std() + 1e-5)
                
                random_distilation_loss = self.random_distillation_network.update(states)
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + random_distilation_loss

                total_loss_buffer.append(total_loss.item())
                actor_loss_buffer.append(actor_loss.item())
                critic_loss_buffer.append(critic_loss.item())


                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                self.random_distillation_network.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                self.random_distillation_network.optimizer.step()

        self.replay_buffer.clear_memory()

        return {"total_loss" : np.mean(total_loss_buffer),
                "actor_loss" : np.mean(actor_loss_buffer),
                "critic_loss" : np.mean(critic_loss_buffer)}

    def get_action(self, obs: np.ndarray):

        action, prob = self.actor.get_action(obs)
        state = torch.tensor(obs, dtype=torch.float32, device=self.device)
        value = self.critic.forward(state)

        return action.cpu().detach().numpy(), prob.cpu().detach().numpy(), value.cpu().detach().numpy()
