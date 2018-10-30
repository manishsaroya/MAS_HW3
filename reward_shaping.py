import numpy as np
import matplotlib.pyplot as plt
import copy

# Number of Agents
numAgents = 30

# Number of Nights 
numNights = 7

# Optimal Number of agents at each night
numOptimalAgents = 5

# learning rate 
alpha = 0.1

# exploration rate
epsilon = 0.1

rewardEstimates = np.zeros((numAgents, numNights))

def takeActions():
    attendance = np.zeros(numNights)
    actions = np.zeros(numAgents)
    #print "attendance", attendance
    for i in range(numAgents):
        if np.random.uniform(0,1) < epsilon:
            action = np.random.randint(0,numNights)
        else:
            action = np.argmax(rewardEstimates[i])
        actions[i] = action
        attendance[action] += 1
    #print "post attendance", attendance
    #print actions
    return attendance,actions

def updateLocalRewardEstimates(attendance,actions):
    #print "in func",actions,attendance
    for agent, action in enumerate(actions):
        #print attendance[int(action)]
        #print (-1 * attendance[int(action)])/numOptimalAgents
        sample = attendance[int(action)] * np.exp((-1 * attendance[int(action)])/numOptimalAgents)
        rewardEstimates[agent][int(action)] = (1-alpha) * rewardEstimates[agent][int(action)] + alpha * sample

def updateGlobalRewardEstimates(attendance,actions):
    #print "in func",actions,attendance
    sample = computeGlobalReward(attendance,actions)
    for agent, action in enumerate(actions):
        #print attendance[int(action)]
        #print (-1 * attendance[int(action)])/numOptimalAgents
        #sample = np.exp((-1 * attendance[int(action)])/numOptimalAgents)
        
        rewardEstimates[agent][int(action)] = (1-alpha) * rewardEstimates[agent][int(action)] + alpha * sample

def counterfactualAction(attendance,actions,agent):
    #print "in loop ", attendance
    attendance[int(actions[agent])] -= 1
    action = np.random.randint(0,numNights)
    actions[agent] = action
    attendance[action] += 1
    #print "inloop post" ,attendance
    return attendance, actions

def counterfactualActionAbsence(attendance,actions,agent):
    #print "in loop ", attendance
    attendance[int(actions[agent])] -= 1
    return attendance, actions

def updateDifferenceRewardEstimates(attendance,actions):
    #print "in func",actions,attendance
    globalSample = computeGlobalReward(attendance,actions)
    for agent, action in enumerate(actions):
        #print attendance[int(action)]
        #print (-1 * attendance[int(action)])/numOptimalAgents
        #sample = np.exp((-1 * attendance[int(action)])/numOptimalAgents)
        #print "pre atten", attendance
        atte, act = counterfactualAction(copy.copy(attendance),copy.copy(actions),agent)
        #print "compare actions",cmp(act,actions)
        #print "compare attendance", cmp(atte,attendance)
        #print "counterfact" ,atte, act
        #print "orig" , attendance ,actions
        sample = globalSample - computeGlobalReward(atte,act)
        #print sample
        rewardEstimates[agent][int(action)] = (1-alpha) * rewardEstimates[agent][int(action)] + alpha * sample

def updateDifferenceRewardEstimatesAbsence(attendance,actions):
    globalSample = computeGlobalReward(attendance,actions)
    for agent, action in enumerate(actions):
        atte, act = counterfactualActionAbsence(copy.copy(attendance),copy.copy(actions),agent)
        sample = globalSample - computeGlobalReward(atte,act)
        rewardEstimates[agent][int(action)] = (1-alpha) * rewardEstimates[agent][int(action)] + alpha * sample

def computeGlobalReward(attendance,actions):
    globalReward = 0
    for i in range(numNights):
        globalReward += attendance[i] * np.exp((-1*attendance[i])/numOptimalAgents)
    return globalReward


# Visualization
fig = plt.figure(figsize=(4, 4))
#im = plt.imshow(, origin={'lower','left'})

if __name__ == '__main__':
    print rewardEstimates
    rewardList = []
    a = []
    for i in range(500):
        atten, actions = takeActions()
        a = atten
        #updateLocalRewardEstimates(atten,actions)
        #updateGlobalRewardEstimates(atten,actions) 
        updateDifferenceRewardEstimates(atten, actions)
        rewardList.append(computeGlobalReward(atten,actions))
        epsilon *= 0.999
        print epsilon
    #print rewardEstimates 
    print rewardList
    print "Difference", a
    plt.plot(rewardList,label='Difference Reward c_i-random action')
    plt.ylabel('Global Reward')
    plt.xlabel('No. of Episodes')

    rewardEstimates = np.zeros((numAgents, numNights))
    epsilon = 0.1

    # Making all plots in one
    rewardList1 = []
    a1 = []
    for i in range(500):
        atten, actions = takeActions()
        a1 = atten
        #updateLocalRewardEstimates(atten,actions)
        updateGlobalRewardEstimates(atten,actions)
        #updateDifferenceRewardEstimates(atten, actions)
        rewardList1.append(computeGlobalReward(atten,actions))
        epsilon *= 0.999
        print epsilon
    print "Global", a1 
    plt.plot(rewardList1,label='Global Reward')


    rewardEstimates = np.zeros((numAgents, numNights))
    epsilon = 0.1

    # Making all plots in one
    rewardList2 = []
    a2 = []
    for i in range(500):
        atten, actions = takeActions()
        a2 = atten
        updateLocalRewardEstimates(atten,actions)
        #updateGlobalRewardEstimates(atten,actions)
        #updateDifferenceRewardEstimates(atten, actions)
        rewardList2.append(computeGlobalReward(atten,actions))
        epsilon *= 0.999
        print epsilon
    print "Local", a2
    plt.plot(rewardList2,label='Local Reward')


    rewardEstimates = np.zeros((numAgents, numNights))
    epsilon = 0.1

    # Making all plots in one
    rewardList3 = []
    a3 = []
    for i in range(500):
        atten, actions = takeActions()
        a3 = atten
        #updateLocalRewardEstimates(atten,actions)
        #updateGlobalRewardEstimates(atten,actions)
        updateDifferenceRewardEstimatesAbsence(atten, actions)
        rewardList3.append(computeGlobalReward(atten,actions))
        epsilon *= 0.999
        print epsilon
    print "Local", a3
    plt.plot(rewardList3,label='Difference Reward c_i-No action')

    # ploting the histogram
    # fixed bin size
    #plt.figure(figsize=(12, 12))
    #bins = np.arange(0, numNights, 1) # fixed bin size

    #plt.xlim([min(a)-5, max(a)+5])

    #plt.hist(a, bins=bins, alpha=0.5)
    #plt.title('Random Gaussian data (fixed bin size)')
    #plt.xlabel('variable X (bin size = 5)')
    #plt.ylabel('count')
plt.legend()    
plt.show()

