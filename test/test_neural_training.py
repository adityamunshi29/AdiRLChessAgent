import numpy as np  
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

from RLC.real_chess import agent, environment, learn, tree
import chess
from chess.pgn import Game

from keras.models import Model, clone_model, load_model

opponent = agent.GreedyAgent() #'greedy agent' issues moves, simply trying to maximise material
env = environment.Board(opponent, FEN=None)
player = agent.Agent(lr=0.001, network='Adi1')
learner = learn.TD_search(env, player, gamma=0.8, search_time=1.5)
node = tree.Node(learner.env.board, gamma=learner.gamma)
player.model.summary()

w_before = learner.agent.model.get_weights()
# print(w_before.shape)

def test_train():
    learner.learn(iters=400, model_name='adi42_')
    reward_smooth = pd.DataFrame(learner.reward_trace)
    reward_smooth.rolling(window=500, min_periods=0).mean().plot(figsize=(16, 9),
                                                                 title='average performance over the last 3 episodes')
    plt.show()

    reward_smooth = pd.DataFrame(learner.piece_balance_trace)
    reward_smooth.rolling(window=100, min_periods=0).mean().plot(figsize=(16, 9),
                                                                 title='average piece balance over the last 3 episodes')
    plt.show()


test_train()

w_after = learner.agent.model.get_weights()
# print(w_after.shape)
print("done")
