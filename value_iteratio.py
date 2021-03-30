import numpy as np
from action import actions
import matplotlib.pyplot as plt

def main():
    print('Initializing parameters....')
    # hyperparameters
    gamma = 0.99  # discount factor 0.984545
    convergence_cutoff = 1e-5 #cut off at smallest difference between old and new utilties
    intention = 0.8  # probability of choosing an intended action
    stray = 0.1  # probability agent move either right angle to intended action

    # Define the maze
    maze_states = []
    for c in range(6):
        for r in range(6):
            maze_states.append((c, r))  # r : x ; c : y

    # define rewards
    rewards = {}
    # define initial value function
    value = {}
    utility_estimates = {}
    #initialize rewards and functions
    for s in maze_states:
        utility_estimates[s] = [0]
        # switch case to assign rewards to each state
        # green squares
        if s == (0, 0) or s == (2, 0) or s == (5, 0) or s == (3, 1) or s == (4, 2) or s == (5, 3):
            rewards[s] = 1.0
            value[s] = 0
        # brown squares
        elif s == (1, 1) or s == (5, 1) or s == (2, 2) or s == (3, 3) or s == (4, 4):
            rewards[s] = -1
            value[s] = 0
        # declare large penalty for entering walls
        elif s == (1,0) or s == (4, 1) or s == (1, 4) or s == (2, 4) or s == (3, 4):
            rewards[s] = -9999
            value[s] = 0
        else:
            rewards[s] = -0.04
            value[s] = 0

    # define dictionary of possible actions (no end states)
    avail_actions = actions

    # define initial policy (create random actions to take at each states)
    policy = {}
    for s in avail_actions.keys():
        tag = {i: a for i, a in zip([i for i in range(len(avail_actions[s]))], avail_actions[s])}
        policy[s] = tag[np.random.choice(list(tag.keys()))]

    print('Start value iteration.....')

    iteration = 1
    while True:
        # print(value)
        print('commencing iteration {}...'.format(iteration))
        biggest_change = 0
        update_value = value.copy() #prevent pass by reference
        for s in maze_states: #iterate through each state in the maze
            if s in list(policy.keys()): #ignore wall states (0 utility)
                old_v = value[s]
                new_v = 0

                '''
                iterate through every single actions at the state to find max utility action
                Each action will have stray actions are either 90 degrees movement
                '''

                for a in avail_actions[s]:
                    if a == 'U':
                        next_state = tuple([s[0] , s[1]-1])
                        stray_next_states = {'L': tuple([s[0]-1, s[1]]), 'R': tuple([s[0]+1, s[1] ])}
                    elif a == 'D':
                        next_state = tuple([s[0] , s[1]+1])
                        stray_next_states = {'L': tuple([s[0]-1, s[1] ]), 'R': tuple([s[0]+1, s[1] ])}
                    elif a == 'L':
                        next_state = tuple([s[0]-1, s[1]])
                        stray_next_states = {'U': tuple([s[0] , s[1]-1]), 'D': tuple([s[0] , s[1]+1])}
                    elif a == 'R':
                        next_state = tuple([s[0]+1, s[1]])
                        stray_next_states = {'U': tuple([s[0], s[1]-1]), 'D': tuple([s[0], s[1]+1])}

                    stationary_prob = 0 #account for no movement prob (hit wall or Out of bounds (OB)
                    prob_checker = 0 #check that total probability is 100%

                    #check if next state is OB or wall (reward at wall is -9999, use this to check if nextstate goes into wall)
                    if next_state[0] < 0 or next_state[0] > 5 or next_state[1] < 0 or next_state[1] > 5 or rewards[next_state] == -9999:
                        stationary_prob += intention
                        next_state = None
                    #check if stray movement is OB or wall
                    for sa,ns in stray_next_states.items():
                        if ns[0] < 0 or ns[0] > 5 or ns[1] < 0 or ns[1] > 5 or rewards[ns] == -9999:
                            stationary_prob += stray
                            stray_next_states[sa] = None
                    #calculate the expectation of action
                    u = 0

                    if next_state != None:
                        u += intention * value[next_state]
                        prob_checker += intention
                    for sa, ns in stray_next_states.items():
                        if ns != None:
                            u += stray * value[ns]
                            prob_checker += stray
                    u += stationary_prob * value[s]
                    prob_checker += stationary_prob

                    assert prob_checker == 1   #prob check

                    v = rewards[s] + gamma * u #calculate the utility of the state

                    if v > new_v: #Keep the highest utility of the state based on action
                        new_v = v
                        policy[s] = a


                #save the best of all actions for the state
                update_value[s] = new_v
                utility_estimates[s].append(new_v)
                biggest_change = max(biggest_change,np.abs(old_v-new_v)) #account for the largest utility update changes
        value = update_value.copy() #save Ut+1 into Ut
        #if update is less than the cut off, convergence has taken place (end algorithm)
        if biggest_change < convergence_cutoff:
            break
        iteration += 1

    print('End of Program....')
    print("coordinates are in (col,row) format with the top left corner being (0,0)\n")

    #flip the format first (assignment is (col,row), code written accidentally in (row,col)
    # new_value = {}
    # for s, v in value.items():
    #     new_value[tuple([s[1],s[0]])] = v

    for s, v in value.items(): #print Utilities
        print("{} : {}".format(s, v))

    #print optimal policy)
    action_dict = {'U': '^', 'D': 'V', 'L': '<', 'R': '>', 'X': 'X'}
    policy[tuple([1, 0])] = 'X'
    policy[tuple([4, 1])] = 'X'
    policy[tuple([1, 4])] = 'X'
    policy[tuple([2, 4])] = 'X'
    policy[tuple([3, 4])] = 'X'
    for c in range(6):
        print("{} {} {} {} {} {}".format(action_dict[policy[tuple([0, c])]], action_dict[policy[tuple([1, c])]],
                                         action_dict[policy[tuple([2, c])]], action_dict[policy[tuple([3, c])]],
                                         action_dict[policy[tuple([4, c])]], action_dict[policy[tuple([5, c])]]))
    print(policy)
    print(iteration)
    #plot the utility estimates per iteration
    for s,ue_list in utility_estimates.items():
        # if s == (1,0) or s == (4,1) or s == (1,4) or s == (2,4) or s == (3,4):
        #     ue_list += [0 for i in range(iteration)]
        if s in avail_actions.keys():

            plt.plot([i for i in range(iteration+1)], list(ue_list))
    plt.legend(list(avail_actions.keys()),loc='lower right',prop={'size': 8},ncol=4)
    plt.xlabel('Iterations')
    plt.ylabel('Utility estimates')
    plt.title("Utility estimates vs iterations")
    plt.show()


main()
