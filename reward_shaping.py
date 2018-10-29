import numpy as np
import matplotlib.pyplot as plt

# Number of Agents
numAgents = 40

# Number of Nights 
numNights = 5

# Optimal Number of agents at each night
numOptimalAgents = 4

# learning rate 
alpha = 0.5

# exploration rate
epsilon = 0.1

rewardEstimates = np.zeros((numAgents, numNights))

def takeActions():
    attendance = np.zeros(numNights)
    actions = np.zeros(numAgents)
    print "attendance", attendance
    for i in range(numAgents):
        if np.random.uniform(0,1) < epsilon:
            action = np.random.randint(0,numNights)
        else:
            action = np.argmax(rewardEstimates[i])
        actions[i] = action
        attendance[action] += 1
    print "post attendance", attendance
    print actions
    return attendance,actions

def updateLocalRewardEstimates(attendance,actions):
    print "in func",actions,attendance
    for agent, action in enumerate(actions):
        #print attendance[int(action)]
        #print (-1 * attendance[int(action)])/numOptimalAgents
        sample = np.exp((-1 * attendance[int(action)])/numOptimalAgents)
        rewardEstimates[agent][int(action)] = (1-alpha) * rewardEstimates[agent][int(action)] + alpha * sample

def updateGlobalRewardEstimates(attendance,actions):
    print "in func",actions,attendance
    sample = computeGlobalReward(attendance,actions)
    for agent, action in enumerate(actions):
        #print attendance[int(action)]
        #print (-1 * attendance[int(action)])/numOptimalAgents
        #sample = np.exp((-1 * attendance[int(action)])/numOptimalAgents)
        rewardEstimates[agent][int(action)] = (1-alpha) * rewardEstimates[agent][int(action)] + alpha * sample

def updateDifferenceRewardEstimates(attendance,actions):
    print "in func",actions,attendance
    for agent, action in enumerate(actions):
        #print attendance[int(action)]
        #print (-1 * attendance[int(action)])/numOptimalAgents
        sample = np.exp((-1 * attendance[int(action)])/numOptimalAgents)
        rewardEstimates[agent][int(action)] = (1-alpha) * rewardEstimates[agent][int(action)] + alpha * sample

def computeGlobalReward(attendance,actions):
    globalReward = 0
    for i in range(numNights):
        globalReward += attendance[i] * np.exp((-1*attendance[i])/numOptimalAgents)
    return globalReward


# Visualization
fig = plt.figure(figsize=(6, 6))
#im = plt.imshow(, origin={'lower','left'})

if __name__ == '__main__':
    print rewardEstimates
    rewardList = []
    for i in range(100):
        atten, actions = takeActions()
        #updateLocalRewardEstimates(atten,actions)
        updateGlobalRewardEstimates(atten,actions)
        rewardList.append(computeGlobalReward(atten,actions))
    print rewardEstimates 
    print rewardList
    plt.plot(rewardList)
    plt.ylabel('Global Reward')
    plt.xlabel('No. of Episodes')
    
plt.show()

