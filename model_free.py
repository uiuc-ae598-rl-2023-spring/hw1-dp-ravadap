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

def SARSA(env, num_episodes, gamma, alpha, epsilon, pendulum):
    # initialize Q matrix
    Q = np.zeros((env.num_states, env.num_actions))

    # rewards for plotting
    ep_reward = np.zeros(num_episodes)

    # loop through episodes
    for episode in range(num_episodes):
        # initialize s
        state = env.reset()
        action = epsilon_greedy_policy(Q, epsilon, state)

        # Create log to store data from simulation
        log = {
            't': [0],
            's': [state],
            'a': [],
            'r': [],
        }

        if pendulum:
            log['theta'] = [env.x[0]] 
            log['thetadot'] = [env.x[1]]

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
            ep_reward[episode] += reward*gamma*t

            log['t'].append(log['t'][-1] + 1)
            log['s'].append(state)
            log['a'].append(action)
            log['r'].append(reward)

            if pendulum:
                log['theta'].append(env.x[0])
                log['thetadot'].append(env.x[1])

            if done: break

    return Q, ep_reward, log

def Q_learning(env, num_episodes, gamma, alpha, epsilon, pendulum):
    # initialize Q matrix
    Q = np.zeros((env.num_states, env.num_actions))

    # rewards for plotting
    ep_reward = np.zeros(num_episodes)

    # loop through episodes
    for episode in range(num_episodes):
        # initialize s
        state = env.reset()

        # Create log to store data from simulation
        log = {
            't': [0],
            's': [state],
            'a': [],
            'r': [],
        }

        if pendulum:
            log['theta'] = [env.x[0]] 
            log['thetadot'] = [env.x[1]]

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
            ep_reward[episode] += reward*gamma*t

            log['t'].append(log['t'][-1] + 1)
            log['s'].append(state)
            log['a'].append(action)
            log['r'].append(reward)

            if pendulum:
                log['theta'].append(env.x[0])
                log['thetadot'].append(env.x[1])

            if done: break

    return Q, ep_reward, log

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

# plotting trajectory
def plotting_sar_trace_model_free(num_episodes, ep_reward, log, path, pendulum):
    # # Initialize simulation
    # s = env.reset()

    # # Create log to store data from simulation
    # log = {
    #     't': [0],
    #     's': [s],
    #     'a': [],
    #     'r': [],
    # }

    # if pendulum:
    #     log['theta'] = [env.x[0]] 
    #     log['thetadot'] = [env.x[1]]

    # # Simulate until episode is done
    # done = False
    # while not done:
    #     a = np.argmax(policy[s])
    #     (s, r, done) = env.step(a)
    #     log['t'].append(log['t'][-1] + 1)
    #     log['s'].append(s)
    #     log['a'].append(a)
    #     log['r'].append(r)

    #     if pendulum:
    #         log['theta'].append(env.x[0])
    #         log['thetadot'].append(env.x[1])

    # Subplots
    fig, axs = plt.subplots(2, 1)
    
    if pendulum: fig, axs = plt.subplots(3, 1)

    # Plot data and save to png file
    axs[0].plot(np.arange(num_episodes), ep_reward)
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Return')
    axs[0].grid()

    axs[1].plot(log['t'], log['s'])
    axs[1].plot(log['t'][:-1], log['a'])
    axs[1].plot(log['t'][:-1], log['r'])
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('State | Action | Reward')
    axs[1].legend(['s', 'a', 'r'], loc="upper right")
    axs[1].grid()

    if pendulum: 
        axs[2].plot(log['t'], log['theta'])
        axs[2].plot(log['t'], log['thetadot'])
        axs[2].set_xlabel('t')
        axs[2].set_ylabel('Theta | ThetaDot')
        axs[2].legend(['Theta', 'ThetaDot'])
        axs[2].grid()
    
    fig.tight_layout()
    
    plt.savefig(path)

# plotting state value function and policy
def policy_and_state_value_function_plotting(policy, V, path):
    fig, axs = plt.subplots(1, 2, figsize=(25, 25))

    # policy
    # policy_actions = ['right', 'up', 'left', 'down']
    axs[0].imshow(policy, interpolation='nearest', cmap='cool')
    axs[0].set_title('Policy')
    for x,y in itertools.product(range(15), range(21)):
        val = policy[x,y]
        axs[0].text(y,x, f'{val}', ha  = 'center', va = 'center', size = 'x-small', weight = 'roman')

    # state value function
    axs[1].imshow(V, interpolation='nearest', cmap='cool')
    axs[1].set_title('State Value Function')
    for x,y in itertools.product(range(15), range(21)):
        val = '{:.2f}'.format(round(V[x,y], 2))
        axs[1].text(y,x, f'{val}', ha  = 'center', va = 'center', size = 'x-small', weight = 'roman')

    plt.savefig(path)

# plotting variation of parameter
def plot_vary_parameter(num_episodes, return_list_epsilon, return_list_alpha, epsilon_list, alpha_list, path):
    # Subplots
    fig, axs = plt.subplots(2, 1)

    ep_legend = []
    alp_legend = []

    for i in range(len(return_list_epsilon)):
        axs[0].plot(np.arange(num_episodes), return_list_epsilon[i])
        ep_legend.append('epsilon = ' + str(epsilon_list[i]))

        axs[1].plot(np.arange(num_episodes), return_list_alpha[i])
        alp_legend.append('alpha = ' + str(alpha_list[i]))

    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Return')
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[0].legend(ep_legend)

    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Return')
    axs[1].set_yscale('log')
    axs[1].grid()
    axs[1].legend(alp_legend)

    plt.savefig(path)

