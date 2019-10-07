# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import talib 

# prepare the data sets
columes = ['open','high','low','close','volume'] #, 'quote_volume', 'num_trade', 'buy_base', 'buy_quote']
dataset_train = pd.read_csv('BTCUSDT.csv' , skipinitialspace=True, usecols=columes)
dataset_train.shape # (1092470, 3)

training_set = dataset_train.iloc[:,:].values # close data
training_set.shape #(1092470, 9)

long = 1430
short = 1060
RSIt = 14
ATRt = 14


close_price = np.array([ i[3] for i in training_set]) # close_price[-1]/close_price[0] # 1.002
low_price   = np.array([ i[2] for i in training_set])
high_price  = np.array([ i[1] for i in training_set])

# append MA_long
MA_long  = talib.MA(close_price, timeperiod=long, matype=0)
MA_long.shape # (801453,)
MA_long = np.reshape(MA_long, (MA_long.shape[0],1))
MA_long.shape # (801453, 1)
training_set = np.append(training_set, MA_long, axis=1)
training_set.shape #(801453, 10)

# append MA_short
MA_short = talib.MA(close_price, timeperiod=short, matype=0)
MA_short.shape # (801453,)
MA_short = np.reshape(MA_short, (MA_short.shape[0],1))
training_set = np.append(training_set, MA_short, axis=1)
training_set.shape #(801453, 11)

# append ATR
ATR      = talib.ATR(high_price ,low_price,close_price, timeperiod=ATRt)
ATR.shape # (801453,)
ATR = np.reshape(ATR, (ATR.shape[0],1))
training_set = np.append(training_set, ATR, axis=1)
training_set.shape #(801453, 12)

# append RSI
RSI      = talib.RSI(close_price, timeperiod=RSIt)
RSI.shape # (801453,)
RSI = np.reshape(RSI, (ATR.shape[0],1))
training_set = np.append(training_set, RSI, axis=1)
training_set.shape #(1092470, 9)

# choose the trainning set
training_set = training_set[int(1092470/20*0):int(1092470/20*3)]
training_set.shape # (546235, 9)
#training_set[-1][3]/training_set[0][3] # 0.78334
close_price = np.array([ i[3] for i in training_set])
plt.plot(close_price)
plt.show()

print('market growth: ', training_set[-1][3]/training_set[0][3])

# normalize the matrix
training_set[np.isnan(training_set)] = 0
for i in range(training_set.shape[1]):
    v = training_set[:,i]
    training_set[:,i] = (v-v.min())/(v.max()-v.min())




'''
a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [10,11,12]])

a.shape #(4,3)


b = np.array([[0],[0],[0],[0]])
b.shape # (4,1)

np.append(a, b, axis=1)
a.shape # (4, 3)

b2 = np.array([0,0,0,0])
b2 = np.reshape(b2,(4,1))  # use reshape to append the parameters
np.append(a, b2, axis=1)
a.shape # (4, 3)

type(training_set) # numpy.ndarray
training_set.shape # (801453, 9)
'''



class env():

    
    def __init__(self,wind, play_speed):
        self.wind = wind
        # get the prices and indicators
        self.price = close_price
        self.prices_indicators = training_set
        
        self.action = None # 0:sell, 1:buy, None:do nothing
        self.money_start = 1000
        self.money = self.money_start
        self.bought_start = self.money_start/self.price[0]
        self.bought = self.bought_start
        self.state = 1430 # wind start index
        self.status = 0 # 0: short, 1: long
        self.trade_fee = 0.999
        self.gameOver = False
        self.lifePenalty = -0.00001
        
        self.play_speed = play_speed
        
        self.rewardList = list()
        self.actionList = list()
        self.moneyList = list()
        self.ini_obs = self.prices_indicators[self.state : self.state + self.wind]
        
    def reset(self):
        
        self.money_start = 1000
        self.money = 1000
        self.bought = 0
        self.action = None
        self.state = 1430 # wind start index
        self.status = 0 # 0: short, 1: long
        self.gameOver = False
        
        #obs = self.prices_indicators[self.state : self.state + self.wind]
        #reward = self.lifePenalty
        self.rewardList = []
        self.actionList = []
        self.moneyList  = []
        '''
        print('bought/bought_start =', self.bought/self.bought_start, 
              'money/money_start =', self.money/self.money_start)
        '''
        #return obs, reward, self.gameOver
    
    
    def step(self, action):
        
        self.actionList.append(action)
        
        if action == 1 and self.status == 0: # buy at short
            
            self.bought = self.money / self.price[self.state+self.wind-1] * self.trade_fee
            self.status = 1
            
            self.state += self.play_speed
            # if the game is over
            if self.state+self.wind >= len(self.price)- self.play_speed -1 :
                self.gameOver = True
            
            obs = self.prices_indicators[self.state:self.state+self.wind]
            reward = (self.price[self.state+self.wind-1]/self.price[self.state-1]) * self.trade_fee - 1               
            
            self.money = self.money * self.trade_fee
            self.moneyList.append(self.money)
            self.rewardList.append(reward)
            '''
            print('bought/bought_start =', self.bought/self.bought_start, 
              'money/money_start =', self.money/self.money_start)
            '''
            return obs, reward, self.gameOver
        
        elif action == 1 and self.status == 1: # buy at long
            
            
            self.state += self.play_speed
            # if the game is over
            if self.state+self.wind >= len(self.price)- self.play_speed -1 :
                self.gameOver = True
            
            obs = self.prices_indicators[self.state:self.state+self.wind]
            reward = (self.price[self.state+self.wind-1]/self.price[self.state-1]) - 1
            
            # append money
            self.money = self.bought * self.price[self.state+self.wind-1]
            self.moneyList.append(self.money)
            self.rewardList.append(reward)
            '''
            print('bought/bought_start =', self.bought/self.bought_start, 
              'money/money_start =', self.money/self.money_start)
            '''
            return obs, reward, self.gameOver
            
        elif action == 0 and self.status == 1: # sell at long
            
            self.money = self.bought * self.price[self.state+self.wind-1] * self.trade_fee
            self.status = 0
            
            self.state += self.play_speed
            # if the game is over
            if self.state+self.wind >= len(self.price)- self.play_speed -1 :
                self.gameOver = True
            
            
            obs = self.prices_indicators[self.state:self.state+self.wind]
            reward = 1 * self.trade_fee - (self.price[self.state+self.wind-1]/self.price[self.state-1])
            
            # append money
            self.moneyList.append(self.money)
            self.rewardList.append(reward)
            '''
            print('bought/bought_start =', self.bought/self.bought_start, 
              'money/money_start =', self.money/self.money_start)
            '''
            return obs, reward, self.gameOver
        
        elif action == 0 and self.status == 0: # sell at short
            
            self.state += self.play_speed
            # if the game is over
            if self.state+self.wind >= len(self.price)- self.play_speed -1 :
                self.gameOver = True
            
            obs = self.prices_indicators[self.state:self.state+self.wind]
            reward = 1  - (self.price[self.state+self.wind-1]/self.price[self.state-1])
            
            # append money
            self.moneyList.append(self.money)
            self.rewardList.append(reward)
            '''
            print('bought/bought_start =', self.bought/self.bought_start, 
              'money/money_start =', self.money/self.money_start)
            '''
            return obs, reward, self.gameOver
        
        else:
            print('error with the action/status condition')
            
        
        



'''
# play in the environment
env1 = env(wind=13)
env1.reset()
obs, reward, gameOver = env1.step(1)
obs.shape # (15, 13)
obs[0]


obs, reward, gameOver = env1.step(1)
'''
















