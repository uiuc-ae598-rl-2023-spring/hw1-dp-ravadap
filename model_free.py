import numpy as np
import itertools
import matplotlib.pyplot as plt

def epsilon_greedy_policy(Q, epsilon, state):
    if np.random.uniform(0,1) < epsilon:
        # choose a random action
        return np.random.randint(Q.shape[1])
    else:
        # choose best action
        return np.argmax(Q[state])

def SARSA(env, num_episodes, gamma, alpha, epsilon, path):
    # initialize Q matrix
    Q = np.zeros((env.num_states, env.num_actions))

    # rewards for plotting
    ep_reward = np.zeros(num_episodes)

    # loop through episodes
    for episode in range(num_episodes):
        # initialize s
        state = env.reset()
        action = epsilon_greedy_policy(Q, epsilon, state)

        # For plotting
        # Create log to store data from simulation
        log = {
            't': [0],
            's': [state],
            'a': [],
            'r': [],
        }

        # step through the environment
        for t in itertools.count():
            # take a step
            next_state, reward, done = env.step(action)

            # choose next action
            next_action = epsilon_greedy_policy(Q, epsilon, next_state)

            # calculate Q value
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            # update state and action
            state = next_state
            action = next_action

            # update values for plotting
            ep_reward[episode] += reward

            log['t'].append(log['t'][-1] + 1)
            log['s'].append(state)
            log['a'].append(action)
            log['r'].append(reward)

            if done: break

    # Subplots
    fig, axs = plt.subplots(2, 1)

    # Plot data and save to png file
    axs[0].plot(np.arange(num_episodes), ep_reward)
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Reward')
    axs[0].grid()

    axs[1].plot(log['t'], log['s'])
    axs[1].plot(log['t'][:-1], log['a'])
    axs[1].plot(log['t'][:-1], log['r'])
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('State | Action | Reward')
    axs[1].legend(['s', 'a', 'r'], loc="upper right")
    axs[1].grid()

    fig.tight_layout()
    
    plt.savefig(path)

    return Q

def Q_learning(env, num_episodes, gamma, alpha, epsilon, path):
    # initialize Q matrix
    Q = np.zeros((env.num_states, env.num_actions))

    # rewards for plotting
    ep_reward = np.zeros(num_episodes)

    # loop through episodes
    for episode in range(num_episodes):
        # initialize s
        state = env.reset()

        # For plotting
        # Create log to store data from simulation
        log = {
            't': [0],
            's': [state],
            'a': [],
            'r': [],
        }

        # step through the environment
        for t in itertools.count():
            # take a step
            action = epsilon_greedy_policy(Q, epsilon, state)
            next_state, reward, done = env.step(action)

            # choose next best action
            best_next_action = np.argmax(Q[next_state])

            # calculate Q value
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

            # update state and action
            state = next_state

            # update values for plotting
            ep_reward[episode] += reward

            log['t'].append(log['t'][-1] + 1)
            log['s'].append(state)
            log['a'].append(action)
            log['r'].append(reward)

            if done: break

    # Subplots
    fig, axs = plt.subplots(2, 1)

    # Plot data and save to png file
    axs[0].plot(np.arange(num_episodes), ep_reward)
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Reward')
    axs[0].grid()

    axs[1].plot(log['t'], log['s'])
    axs[1].plot(log['t'][:-1], log['a'])
    axs[1].plot(log['t'][:-1], log['r'])
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('State | Action | Reward')
    axs[1].legend(['s', 'a', 'r'], loc="upper right")
    axs[1].grid()

    fig.tight_layout()
    
    plt.savefig(path)

    return Q

def TD_0(env, policy, num_episodes, gamma, alpha):
    # intialize value array
    V = np.zeros(env.num_states)

    # looping through all episodes
    for episode in range(num_episodes):
        # initialize state
        state = env.reset()

        # step through the environment
        for t in itertools.count():
            # take an action
            action = np.argmax(policy[state])
            
            # take a step
            next_state, reward, done = env.step(action)

            # update value function 
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])

            # update state
            state = next_state

            if done: break

    return V

