# RL-in-trading
Reinforcement Learning in Trading

By building up the trading environment in the env.py file, which has the same functions of OpenAI gym.

And building up the Reinforcement Learning model including memory-replay, greedy exploration and temporal difference.

The observation is the combining of different indicators calculated from OHLC price as a matrix input into the Convolutional
Neural Network to make buy-sell dicisions in the trading environment, and the reward is the loss of money of gain of money in
each epidode played in the traidng enviroment(the episode is the fixed time period set before trading).

The training result has some overfittig problem which is very obvious, but the further imporvemment has a great potential, this
code can be used as a basic model.
