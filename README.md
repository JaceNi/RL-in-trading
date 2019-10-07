# RL-in-trading
Reinforcement Learning in Trading

By building up the trading environment in the env.py file, which has the same functions of OpenAI gym.

Implementing the Reinforcement Learning model including memory-replay, greedy exploration and temporal difference, which is 
included in the file of brain.py, dqn.py, train.py.

The observation is the combination of different indicators calculated from OHLC price as a matrix input into the Convolutional
Neural Network to make buy-sell dicisions in the trading environment, and the reward is the loss and gain of profit in
each epidode played in the traidng enviroment.

The training result has some overfittig problems which is very obvious, but the further imporvemment has a great potential, and this code can be used as a basic model to improve.
