from RLC.real_chess import agent, environment, learn, tree

from keras.models import Model, clone_model, load_model
import numpy as np
import os.path as path
from chessboard import display


def make_move(env, agent):
    max_move = None
    max_value = np.NINF  # alternatively use really small number-- ensure that this is as small as possible in order to increment maxvalue with r/ to time
    white_move = env.board.turn
    for move in env.board.generate_legal_moves():
        move_str1 = env.board.san(move)
        _, reward = env.step(move)

        if ( env.board.result() == "0-1" and not white_move ) or (white_move and env.board.result() == "1-0"):
            max_move = move
            env.board.pop()
            env.init_layer_board()
            break
        # this rewards draw, needs to change as draw not always the best outcome
        elif env.board.result() == "1/2-1/2":
            max_move = move
            env.board.pop()
            env.init_layer_board()
            break
        else:
            score = agent.predict(
                # not material score. this is score for the move
                np.expand_dims(env.layer_board, axis=0)) + reward
            if score > max_value:
                max_move = move
                max_value = score
            env.board.pop()
            env.init_layer_board()
    move_str = env.board.san(max_move) #checks if move is legal here
    end, _ = env.step(max_move)

    return end, move_str


def play(white, black):
    episode_end = False
    half_move = 1
    while not episode_end and half_move < 80:

        episode_end, latest_move = make_move(env, white if env.board.turn else black)
        # print(str(half_move) + ':'+ latest_move)
        if env.board.turn:
            half_move += 1
    score = env.board.result()
    if score == "0-1":
        return 0.0, 1.0
    elif score == "1-0":
        return 1.0, 0.0
    else:
        return 0.5, 0.5


model1 = load_model(path.join('models', '../models/adi32_2'))
model2 = load_model(path.join('models', '../models/adi_model_0/adi32_3'))
black = agent.GreedyAgent()
env = environment.Board(black, FEN=None)
agent1 = agent.Agent(lr=0.001, network='big', model=model1)
agent2 = agent.Agent(lr=0.001, network='big', model=model2)
agent1.fix_model()
agent2.fix_model()

# test

num_games=10
score1=0
score2=0
while num_games:
    print('games left: '+str(num_games))
    if num_games%2:
        # agent1 is white
        s1, s2 = play(agent1, agent2)
    else:
        # agent2 is white
        s2, s1 = play(agent2, agent1)
    env.board.reset_board()
    score1+=s1
    score2+=s2
    num_games-=1
    print('score, agent1-agent2: ' + str(score1) + '-' + str(score2))

print('final score, agent1-agent2: ' + str(score1) + '-'+str(score2))