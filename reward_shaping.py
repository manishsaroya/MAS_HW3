import numpy as np

# Number of Agents
numAgents = 2

# Number of Nights 
numNights = 2

# Optimal Number of agents at each night
numOptimalAgents = 1

# learning rate 
alpha = 0.5

# exploration rate
epsilon = 0.3

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


rewardEstimates = np.zeros((numAgents, numNights))

def updateRewardEstimates(attendance,actions):
    print "in func",actions,attendance
    for agent, action in enumerate(actions):
        #print attendance[int(action)]
        #print (-1 * attendance[int(action)])/numOptimalAgents
        sample = np.exp((-1 * attendance[int(action)])/numOptimalAgents)
        rewardEstimates[agent][int(action)] = (1-alpha) * rewardEstimates[agent][int(action)] + alpha * sample
        

    
# Visualization
#fig = plt.figure(figsize=(12, 12))
#im = plt.imshow(, origin={'lower','left'})


if __name__ == '__main__':
    print rewardEstimates
    for i in range(100):
        atten, actions = takeActions()
        updateRewardEstimates(atten,actions)
    print rewardEstimates 
    

