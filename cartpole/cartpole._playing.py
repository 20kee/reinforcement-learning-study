import gym
import keyboard

env = gym.make("CartPole-v1", render_mode='human')
env.reset()

reward = 0
while True:
    key = keyboard.read_key()
    print(key)
    if key == 'right' or key == 'left':
        reward += 1
        action = 1 if key == 'right' else 0
        if action in (0, 1):
            env.step(action)
            env.render()
        print(reward)