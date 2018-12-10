import gym 
import numpy as np 
import matplotlib.pyplot as plt  

def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0 
    counter = 0 

    for _ in range(100):
        action = 0 if np.matmul(parameters, observation) < 0 else 1 
        observation, reward , done, info = env.step(action)
        total_reward += reward
        counter +=1 

        if done:
            break
    return total_reward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env = gym.wrappers.Monitor(env, 'cartpole-hill/', force=True)

    bestreward = 0
    counter = 0
    parameters = np.random.rand(4) * 2 - 1 
    noise_scalling = 0.1
    episode_per_update = 5 

    for _ in range(500):
        counter +=1 
        new_parameters = parameters + (np.random.rand(4) * 2 - 1) * noise_scalling
        # print newparams
        # reward = 0
        # for _ in xrange(episodes_per_update):
        #     run = run_episode(env,newparams)
        #     reward += run
        reward = run_episode(env,new_parameters)
        if reward > bestreward:
            bestreward = reward
            parameters = new_parameters
            if reward ==200:
                break
        
    
    if submit:
        for _ in range(100):
            run_episode(env, parameters)
    return counter, bestreward

r, b = train(submit=True)
print(r, b)