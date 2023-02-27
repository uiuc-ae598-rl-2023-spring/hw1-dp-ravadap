import numpy as np

def policy_eval(policy, env, gamma, theta=0.00001):
    # Initialize value array
    V = np.zeros(env.num_states)

    # Loop until delta < theta
    delta = 1
    while delta > theta:
        delta = 0
        # loop for each state to perform a full back up
        for s in range(env.num_states):
            v = 0
            # for all actions
            for a, action_prob in enumerate(policy[s]):
                # all transitions from current state
                for s1 in range(env.num_states):
                    # Successor state cannot be the same as the current state
                    transition_prob = env.p(s1,s,a)
                    if s!= s1 and transition_prob!=0:
                        reward = env.r(s,a)
                        v += action_prob * transition_prob * (reward + gamma * V[s1])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
    return np.array(V)

# helper function to look one step ahead
def look_one_step_ahead(env, gamma, s, V):
    # array to hold expected value of each action
    action_expected_value = np.zeros(env.num_actions)
    for a in range(env.num_actions):
        for s1 in range(env.num_states):
            # Successor state cannot be the same as the current state
            transition_prob = env.p(s1,s,a)
            if s!= s1 and transition_prob!=0:
                reward = env.r(s,a)
                action_expected_value[a] += transition_prob * (reward + gamma * V[s1])
    return action_expected_value

def policy_improvement(env, gamma):

    # random policy
    policy = np.ones((env.num_states, env.num_actions)) / env.num_actions
    mean_value_function = []
    while True:
        policy_stable = True
        
        # current policy
        V = policy_eval(policy, env, gamma)
        
        # loop through for each state
        for s in range(env.num_states):
            # best action under current policy
            old_action = np.argmax(policy[s])

            action_expected_value = look_one_step_ahead(env, gamma, s, V)

            # choose the best action
            best_action = np.argmax(action_expected_value)

            # greedy update 
            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.num_actions)[best_action]

        mean_value_function.append(np.mean(V))
        # return policy if the policy is stable
        if policy_stable: return policy, V, mean_value_function

def value_iteration(env, gamma, theta=0.00001):
    # initialize value array
    V = np.zeros(env.num_states)

    # Loop until delta < theta
    delta = 1
    mean_value_function = []
    while delta > theta:
        delta = 0
        # loop for each state to perform a full back up
        for s in range(env.num_states):
            # get expected values for all possible actions from state
            action_expected_value = look_one_step_ahead(env, gamma, s, V)
            
            # best action value
            best_action_value = np.max(action_expected_value)

            # calculate delta
            delta = max(delta, np.abs(best_action_value - V[s]))

            # update value function
            V[s] = best_action_value

            # calculate mean value
        mean_value_function.append(np.mean(V))

    # Output deterministic policy
    policy = np.zeros((env.num_states, env.num_actions))
    for s in range(env.num_states):
        action_expected_value = look_one_step_ahead(env, gamma, s, V)

        # choose the best action
        best_action = np.argmax(action_expected_value)

        policy[s, best_action] = 1.0

    
    return policy, V, mean_value_function