import numpy as np
import gym
from ppo import Agent
from tqdm import tqdm

def train():

    env = gym.make("CartPole-v1")
    N = 20
    batch_size = 16
    n_epochs = 5
    learning_rate = 3e-4
    agent = Agent(env.action_space.n,
                  input_dims=env.observation_space.shape[0], n_epochs=n_epochs)

    n_games = 300

    score_history = []

    for i in tqdm(range(n_games)):

        observation, _ = env.reset()
        done = False
        score = 0
        learn_iters = 0
        n_steps = 0

        while not done:

            prob, action, value = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            n_steps += 1
            agent.remember(observation, action, prob, value, reward, done)

            if n_steps % N == 0:

                agent.train()
                learn_iters += 1

            observation = observation_
            score_history.append(score)

        print(
            {"game number": n_games,
             "score": np.mean(score_history),
            }
        )

if __name__ == "__main__":
    
    train()
