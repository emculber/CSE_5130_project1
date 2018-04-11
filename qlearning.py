"""
Q-Learning example using OpenAI gym MountainCar enviornment
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
from bridge import Bridge

log = 'debug'

np.set_printoptions(threshold=np.inf)

screen_size_x = 256
screen_size_y = 224
n_states = 40
iter_max = 1000000
current_frame = 0

initial_lr = 1.0 # Learning rate
min_lr = 0.003
gamma = 1.0
t_max = 10000
eps = 0.02

inputs = ['up', 'down', 'left', 'right']

bridge = Bridge()

def run_episode(policy=None, render=False):
    # obs = env.reset() # TODO: Reset state
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

if __name__ == '__main__':
    bridge.connectToSocket()
    np.random.seed(0)
    print ('----- using Q Learning -----')
    q_table = np.zeros(screen_size_x, screen_size_y, 4))
    for i in range(iter_max):
        bridge.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
        for j in range(t_max):
            x, y = bridge.cord()
            x = int(x)
            y = int(y)
            print("(" + str(x) + "," + str(y) + "): " + str(q_table[x][y]))
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(inputs)
                #print("Getting Random Action: " + action)
            else:
                logits = q_table[x][y]
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(inputs, p=probs)
                #print("Getting Set Action: " + action + " With Probabilities: " + str(probs))
            x_, y_, screen, reward, done = bridge.step(action)
            total_reward += reward
            if done:
                reward = -100
            # update q table
            action_n = inputs.index(action)
            # print("Action: " + action + "(" + str(action_n) + ")")
            print("Before Set: " + str(x) + ", " + str(y) + " = " + str(q_table[x][y]) + " (" + str(q_table[x][y][action_n]) + ")")

            print("Setting " + action + " (" + str(action_n) + ") = " + str(reward))
            if(x != x_): 
                # q_table[x][y][action_n] = q_table[x][y][action_n] + eta * (reward + gamma *  np.max(q_table[x_][y_]) - q_table[x][y][action_n])
                q_table[x][y][action_n] = q_table[x][y][action_n] + reward
            elif(y != y_):
                # q_table[x][y][action_n] = q_table[x][y][action_n] + eta * (reward + gamma *  np.max(q_table[x_][y_]) - q_table[x][y][action_n])
                q_table[x][y][action_n] = q_table[x][y][action_n] + reward
            # else:
                # q_table[x][y][action_n] = q_table[x][y][action_n] - 10


            print("After Set: " + str(x) + ", " + str(y) + " = " + str(q_table[x][y]) + " (" + str(q_table[x][y][action_n]) + ")")

            current_frame += 1
            if done:
                break
        print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
        # if i % 100 == 0:
            # print(q_table)
    solution_policy = np.argmax(q_table, axis=2)
    #solution_policy_scores = [run_episode(solution_policy, False) for _ in range(100)]
    #print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    #run_episode(env, solution_policy, True)
