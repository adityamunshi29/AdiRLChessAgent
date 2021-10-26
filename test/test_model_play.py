from RLC.real_chess import agent, environment, learn, tree
import chess
from chess.pgn import Game

from keras.models import Model, clone_model, load_model
import numpy as np
import os.path as path
from chessboard import display
from time import sleep
import chess.svg

def make_move(env, agent, colour=1):
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
                np.expand_dims(env.layer_board, axis=0))*colour
            if score > max_value:
                max_move = move
                max_value = score
            env.board.pop()
            env.init_layer_board()
    move_str = env.board.san(max_move) #checks if move is legal here
    end, _ = env.step(max_move)

    return end, move_str



model = load_model(path.join('models', 'adi42_4'))
black = agent.GreedyAgent()
env = environment.Board(black, FEN=None)
white = agent.Agent(lr=0.001, network='big', model=model)
white.fix_model()


episode_end=False
half_move=1
display.start(env.board.fen())
# chess.svg.board(env.board, size=350)
while not episode_end and half_move<100  and not display.checkForQuit():

    episode_end, latest_move = make_move(env, white if env.board.turn else black, colour=1 if env.board.turn else -1)
    # print(str(half_move) + ": " + latest_move)
    if env.board.turn:
        half_move+=1
    # chess.svg.board(env.board, size=350)
    display.update(env.board.fen())
    # print('--------------------')
    # print(env.board)
    sleep(1)
display.terminate()


pgn = Game.from_board(env.board)
with open("rlc_pgn","w") as log:
    log.write(str(pgn))
    print(str(pgn))