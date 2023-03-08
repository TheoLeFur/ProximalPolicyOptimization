import numpy as np
import gym
import torch
import argparse
from agents.ppo_agent import PPOAgent
from tqdm import tqdm


def train_agent():

    env = gym.make("CartPole-v0")
    N = 20
    n_epochs = 10
    learning_rate = 3e-4

    params = {
        "action_dim": env.action_space.n,
        "observation_dim": env.observation_space.shape[0],
        "n_layers": 2,
        "size": 64,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "learning_rate": 3e-4,
        "discrete": True,
        "eps_clip": 0.1,
    }

    agent = PPOAgent(hparams=params)

    n_games = 300

    score_history = []

    for i in tqdm(range(n_games)):

        observation, _ = env.reset()
        done = False
        score = 0
        learn_iters = 0
        n_steps = 0

        while not done:

            action, prob, value = agent.get_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            n_steps += 1
            agent.add_to_replay_buffer(observation, action, prob, value, reward, done)

            if n_steps % N == 0:
                print("Policy is updating -------")
                agent.train()
                learn_iters += 1
            observation = observation_
            score_history.append(score)

        print(
            {"game number": i,
             "score": np.mean(score_history),
             }
        )


if __name__ == "__main__":
    train_agent()
