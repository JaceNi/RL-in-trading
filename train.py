#Snake: Deep Convolutional Q-Learning - Training file

#Importing the libraries
from env import env
from brain import Brain
from dqn import Dqn
import numpy as np
import matplotlib.pyplot as plt

#Defining the parameters
learningRate = 0.00001
maxMemory = 100000
gamma = 0.9
batchSize = 2
nLastStates = 2

epsilon = 1.
epsilonDecayRate = 0.00002
minEpsilon = 0.

filepathToSave = 'model4.h5'

windSize = 9
play_speed = 15

#Initializing the Environment, the Brain and the Experience Replay Memory 
env = env(windSize, play_speed)
brain = Brain((env.wind,env.prices_indicators.shape[1], nLastStates), learningRate) # colums, rows, num_imgs
#model =  brain.loadModel(filepathToSave)
model = brain.model
DQN = Dqn(maxMemory, gamma)

#Building a function that will reset current state and next state
def resetStates():
     
     currentState = np.zeros((1, env.wind, env.prices_indicators.shape[1], nLastStates))
     
     for i in range(nLastStates):
          currentState[0, :, :, i] = env.ini_obs
     
     return currentState, currentState

#Starting the main loop
epoch = 0
nCollected = 0
total_profit = 0
max_profits = 1.4
scores = list()

while True:
     epoch += 1
     
     #Resetting the Evironment and starting to play the game
     env.reset()
     currentState, nextState = resetStates()
     # currentState.shape # (1, 13, 13, 2)
     # nextState.shape # (1, 13, 13, 2)
     gameOver = False
     while not gameOver:
          #Selecting an action to play
          if np.random.rand() <= epsilon:
               action = np.random.randint(0, 2)
          else:
               qvalues = model.predict(currentState)[0]
               action = np.argmax(qvalues)
          
          #Updating the epsilon and saving the model
          epsilon -= epsilonDecayRate
          epsilon = max(epsilon, minEpsilon)

          #Updating the Environment
          frame, reward, gameOver = env.step(action)
          
          if frame.shape[0] != 9:
              pass
          else:

              # increase the signal of pos reward
              if reward > 0:
                  reward = reward * 1000
              
              # frame.shape # (13,13)
              frame = np.reshape(frame, (1,env.wind, env.prices_indicators.shape[1], 1))
              # frame.shape # (1, 13, 13, 1)
              nextState = np.append(nextState, frame, axis = 3)
              # nextState.shape # (1, 13, 13, 3)
              nextState = np.delete(nextState, 0, axis = 3)
              # nextState.shape # (1, 13, 13, 2)
              
              #Remembering new experience and training the AI
              DQN.remember([currentState, action, reward, nextState], gameOver)
              
              inputs, targets = DQN.getBatch(model, batchSize)
              model.train_on_batch(inputs, targets)
              
              currentState = nextState
              if env.state%15000==0:
                  print(env.state/env.prices_indicators.shape[0],  end = ' ')

     
     # append score in each episode
     total_profit = env.money/env.money_start
     scores.append(total_profit)
     
     if total_profit > max_profits:
          model.save(filepathToSave)
          max_profits = total_profit
     
     # Display the reward
     print('Display the reward ')
     rewardList = env.rewardList
     plt.plot(rewardList)
     plt.xlabel('Episode')
     plt.ylabel('rewardList')
     plt.show()
     
     # Display the actions taken
     print('Display the actions taken ')
     actionLIst = env.actionList
     plt.plot(actionLIst)
     plt.xlabel('Episode')
     plt.ylabel('actionLIst')
     plt.show()
     
     # Display each Episode result   
     print('Display each Episode result ')
     money_list = env.moneyList
     plt.plot(money_list)
     plt.xlabel('Episode')
     plt.ylabel('Money_List')
     plt.show()
     
     #Displaying the improvement
     print('Display the improvement')
     plt.plot(scores)
     plt.xlabel('Epoch')
     plt.ylabel('Total_Profit')
     plt.show()
     
     print('Epoch: ' + str(epoch) + ' Current Best: ' + str(max_profits) + ' Epsilon: {:.5f}'.format(epsilon))
     
     
     
     
     


