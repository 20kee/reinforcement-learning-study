import gym
import time
import numpy as np
import math
import bisect
import threading
from collections import deque


class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode='none')
        self.name = ''
        self.rewards = deque([])

        self.cart_position = [p/10 for p in range(-40, 41)]
        self.cart_velocity = [v/100  for v in range(-130, 135, 5)]
        self.pole_angle = [a/100 for a in range(-22, 23)]
        self.pole_angluar_velocity = [v/10 for v in range(-13, 14)]
        self.Q = np.random.rand(len(self.cart_position), len(self.cart_velocity), len(self.pole_angle), len(self.pole_angluar_velocity), 2)

        self.policy = np.ones((len(self.cart_position), len(self.cart_velocity), len(self.pole_angle), len(self.pole_angluar_velocity)))
        
        for cp in range(len(self.cart_position)):
            print(cp)
            for cv in range(len(self.cart_velocity)):
                for pa in range(len(self.pole_angle)):
                    for pav in range(len(self.pole_angluar_velocity)):
                        self.policy[cp, cv, pa, pav] = np.argmax(self.Q[cp, cv, pa, pav, :])
        
        self.actions = [0, 1]
        self.action_length = 2

        self.epsilon = 0.1
        self.gamma = 0.9
        self.alpha = 0.5

    def real_to_discrete(self, cp, cv, pa, pav):
        return (min(80, bisect.bisect_left(self.cart_position, cp)), min(26, bisect.bisect_left(self.cart_velocity, cv)), min(44, bisect.bisect_left(self.pole_angle, pa)), min(26, bisect.bisect_left(self.pole_angluar_velocity, pav)))

    def make_episode(self, epsilon=0.1):
        T = []
        observation = self.env.reset()
        for t in range(500):
            converted_obs = self.real_to_discrete(*observation) if len(observation) == 4 else self.real_to_discrete(*observation[0])
            action = int(self.policy[converted_obs] if np.random.rand() <= 1-epsilon else np.random.choice(self.actions))
            observation, reward, terminated, truncated, info = self.env.step(action)
            T.append((converted_obs, action, reward))
            if terminated:
                break
        return T
    
    def run_episode(self, T, learn_type):
        if learn_type == 'mc':
            G = 0
            for t in range(len(T)-1, -1, -1):
                state = T[t][0]
                action = T[t][1]
                G = self.gamma*G + T[t][2]
                self.Q[*state, action] += self.alpha * (G - self.Q[*state, action])
                self.policy[*state] = np.argmax(self.Q[*state, :])
        elif learn_type == 'td':
            q = 0
            for t in range(len(T)-1, -1, -1):
                state = T[t][0]
                action = T[t][1]
                r = T[t][2]
                G = r + self.gamma * q
                self.Q[*state, action] += self.alpha * (r +  - self.Q[*state, action])
                q = self.Q[*state, action]
                self.policy[*state] = np.argmax(self.Q[*state, :])
        
    def test_episode(self, env, epsilon=0.0):
        T = []
        r = 0
        observation = env.reset()
        for t in range(500):
            converted_obs = self.real_to_discrete(*observation) if len(observation) == 4 else self.real_to_discrete(*observation[0])
            action = int(self.policy[converted_obs] if np.random.rand() <= 1-epsilon else np.random.choice(self.actions))
            observation, reward, terminated, truncated, info = env.step(action)
            r += reward
            T.append((converted_obs, action))
            if terminated:
                break
        return T
    
    def learn(self):
        episode = 10000
        env_test = gym.make('CartPole-v1', render_mode='human')
        tests = []
        total_reward = 0
        for i in range(5000):
            T = self.make_episode()
            total_reward += len(T)
            self.run_episode(T)

        print("Now model test time !!")
        rewards = []
        for i in range(1000):
            T = self.make_episode(epsilon=0)
            rewards.append(len(T))
        print(rewards)
        print("1000번 시도의 평균 보상은 바로 {} !".format(sum(rewards)//1000))
    

        for i in range(50000):
            T = self.make_episode()
            total_reward += len(T)
            self.run_episode(T)

        print("Now model test time !!")
        rewards = []
        for i in range(1000):
            T = self.make_episode(epsilon=0)
            rewards.append(len(T))
        print(rewards)
        print("1000번 시도의 평균 보상은 바로 {} !".format(sum(rewards)//1000))
    

class MonteCartPole(CartPole):
    def __init__(self):
        super().__init__()
        self.name = 'MC'

    def run_episode(self, T):
        G = 0
        for t in range(len(T)-1, -1, -1):
            state = T[t][0]
            action = T[t][1]
            G = self.gamma*G + T[t][2]
            self.Q[*state, action] += self.alpha * (G - self.Q[*state, action])
            self.policy[*state] = np.argmax(self.Q[*state, :])


class TemporalDiffCartPole(CartPole):
    def __init__(self):
        super().__init__()
        self.name = 'TD'

    def run_episode(self, T):
        q = 0
        for t in range(len(T)-1, -1, -1):
            state = T[t][0]
            action = T[t][1]
            r = T[t][2]
            G = r + self.gamma * q
            self.Q[*state, action] += self.alpha * (r +  - self.Q[*state, action])
            q = self.Q[*state, action]
            self.policy[*state] = np.argmax(self.Q[*state, :])
    





if __name__ == "__main__":
    mc_cartpole = MonteCartPole()

    mc_cartpole.learn()
    