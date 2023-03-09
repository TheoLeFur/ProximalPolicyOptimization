import numpy as np
import gym
import torch
import argparse
from collections import OrderedDict
from agents.ppo_agent import PPOAgent
from tqdm import tqdm


def train_agent(train_params):

    env = gym.make(train_params["env"])
    N = train_params["N"]
    n_epochs = train_params["n_epochs"]
    learning_rate = train_params["learning_rate"]
    n_games = train_params["n_games"]

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
    logs = OrderedDict()
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

                print(" ------- policy is updating - ------")
                agent.train()
                print(agent.train())
                learn_iters += 1
            observation = observation_
            score_history.append(score)

        print(
            {"game number": i,
             "score": np.mean(score_history),
             }
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Proximal Policy Optimization")

    parser.add_argument("env", type = str)
    parser.add_argument("N", type = int, default = 20)
    parser.add_argument("n_epochs", type = int, default = 10)
    parser.add_argument("learning_rate", type = float, default = 3e-4)
    parser.add_argument("n_games", type = int, default = 500)


    args = parser.parse_args()
    params = vars(args)

    train_agent(params)

    









    