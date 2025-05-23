"""
The 10 armed testbed based on Reinforcement Learning : An Introduction by Sutton and Barto.
"""

import numpy as np


class Testbed(object):
    # Constructor
    def __init__(self, nArms, mean, stDev):
        # Number of arms
        self.nArms = nArms

        # Used for the Gaussian random actions
        self.mean = mean  # Mean
        self.stDev = stDev  # Standard Deviation

        self.actArr = np.zeros(nArms)  # Array to store action values
        self.optim = 0  # Store optimal value for greedy
        self.reset()

    # Reset testbed for next iteration
    def reset(self):
        # Set random gaussian/normal values using numpy function, requires mean, standard deviation and number of arms
        self.actArr = np.random.normal(self.mean, self.stDev, self.nArms)

        # Identify the maximum value in action array
        self.optim = np.argmax(self.actArr)


################################################################
# Agent Class - Controls the agents movement and behaviour in the environment interacting with the testbed
# and receives information on the current position
class Agent(object):
    # Constructor
    def __init__(self, nArms, eps=0):
        self.nArms = nArms  # Number of arms
        self.eps = eps

        self.timeStep = 0  # Time Step t
        self.lastAction = None  # Store last action

        self.kAction = np.zeros(nArms)  # count of actions taken at time t
        self.rSum = np.zeros(nArms)  # Sums number of rewards
        self.valEstimates = np.zeros(
            nArms
        )  # action value estimates sum(rewards)/Amount

    # Return string for graph legend
    def __str__(self):
        if self.eps == 0:
            return "Greedy"
        else:
            return "Epsilon = " + str(self.eps)

    # Selects action based on a epsilon-greedy behaviour,
    # if epsilon equals zero, then the agent performs a greedy selection
    def action(self):
        ### POLICY ###
        # Epsilon method
        randProb = np.random.random()  # Pick random probability between 0-1
        if randProb < self.eps:
            a = np.random.choice(len(self.valEstimates))  # Select random action

        # Greedy Method
        else:
            maxAction = np.argmax(self.valEstimates)  # Find max value estimate
            # identify the corresponding action, as array containing only actions with max
            action = np.where(self.valEstimates == np.argmax(self.valEstimates))[0]

            # If multiple actions contain the same value, randomly select an action
            if len(action) == 0:
                a = maxAction
            else:
                a = np.random.choice(action)

        # save last action in variable, and return result
        self.lastAction = a
        return a

    # Interpreter - updates the value extimates amounts based on the last action
    def interpreter(self, reward):
        # Add 1 to the number of action taken in step
        At = self.lastAction

        self.kAction[At] += 1  # Add 1 to action selection
        self.rSum[At] += reward  # Add reward to sum array

        # Calculate new action-value, sum(r)/ka
        self.valEstimates[At] = self.rSum[At] / self.kAction[At]

        # Increase time step
        self.timeStep += 1

    # Reset all variables for next iteration
    def reset(self):
        self.timeStep = 0  # Time Step t
        self.lastAction = None  # Store last action

        self.kAction[:] = 0  # count of actions taken at time t
        self.rSum[:] = 0
        self.valEstimates[:] = 0  # action value estimates Qt ~= Q*(a)


################################################################
# Environment class to control all objects (agent/Testbed)
class Environment(object):
    # Constructor
    def __init__(self, testbed, agents, plays, iterations):
        self.testbed = testbed
        self.agents = agents

        self.plays = plays
        self.iterations = iterations

    # Run Test
    def play(self):
        # Array to store the scores, number of plays X number of agents
        scoreArr = np.zeros((self.plays, len(self.agents)))
        # Array to maintain optimal count, Graph 2
        optimlArr = np.zeros((self.plays, len(self.agents)))

        # loop for number of iterations
        for iIter in range(self.iterations):
            # Print statement after every 100 iterations
            if (iIter % 100) == 0:
                print("Completed Iterations: ", iIter)

            # Reset testbed and all agents
            self.testbed.reset()
            for agent in self.agents:
                agent.reset()

            # Loop for number of plays
            for jPlays in range(self.plays):
                agtCnt = 0

                for kAgent in self.agents:
                    actionT = kAgent.action()

                    # Reward - normal dist (Q*(at), variance = 1)
                    rewardT = np.random.normal(self.testbed.actArr[actionT], scale=1)

                    # Agent checks state
                    kAgent.interpreter(reward=rewardT)

                    # Add score in arrary, graph 1
                    scoreArr[jPlays, agtCnt] += rewardT

                    # check the optimal action, add optimal to array, graph 2
                    if actionT == self.testbed.optim:
                        optimlArr[jPlays, agtCnt] += 1

                    agtCnt += 1

        # return averages
        scoreAvg = scoreArr / self.iterations
        optimlAvg = optimlArr / self.iterations

        return scoreAvg, optimlAvg
