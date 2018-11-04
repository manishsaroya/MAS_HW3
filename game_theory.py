import numpy as np
import matplotlib.pyplot as plt
#learning rate
learning_rate = 0.01

# exploration
epsilon = 0.1

rewardmatrix = np.array([[(1,3),(0,0)],[(0,0),(3,1)]])

print rewardmatrix

player1ClodProb = 0.5

player2ActionEstimates = np.zeros(2)

print player2ActionEstimates

def getReward(actionP1, actionP2):
    reward = rewardmatrix[actionP1,actionP2]
    return reward

def takeActions():
    actions = []
    if np.random.uniform(0,1) < player1ClodProb:
        actions.append(0) # action clod 
    else:
        actions.append(1) # action Mc
    
    if np.random.uniform(0,1) < epsilon:
        actions.append(np.random.randint(0,2))
    else:
        actions.append(np.argmax(player2ActionEstimates))
    #print actions
    return actions


    
if __name__=='__main__':
    rewardList = []
    fig = plt.figure(figsize=(4, 4))
    for k in range(10000):
        actions = takeActions()
        reward = getReward(actions[0],actions[1])
        player2ActionEstimates[actions[1]] = (1 - learning_rate) * player2ActionEstimates[actions[1]] + learning_rate * reward[1]
        rewardList.append(reward[1])
        #epsilon *= 0.999
    
    plt.plot(rewardList,label='leaning')
    plt.ylabel('Reward')
    plt.xlabel('No. of Episodes')
    
    
    # testing phase
    epsilon = 0
    rewardList2 = []
    fig2 = plt.figure(figsize=(4, 4))
    for k in range(5000):
        actions = takeActions()
        reward = getReward(actions[0],actions[1])
        rewardList2.append(reward[1])

    print "average testing reward ", sum(rewardList2) / float(len(rewardList2))
    plt.plot(rewardList2,label='testing')
    plt.ylabel('Reward')
    plt.xlabel('No. of Episodes')
    plt.legend()
    plt.show()
