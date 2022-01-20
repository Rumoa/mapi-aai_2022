import ataxx
import ataxx.players
import time
import numpy as np
import joblib
from minimax_utils import select_depth, easy, medium, pro


# Antonio Ruiz Molero, 2022

players = [ataxx.players.greedy]


n_games = 100

player_1 = [[easy, "easy"], [medium, "medium"], [pro, "pro"]]
player_2 = [[easy, "easy"], [medium, "medium"], [pro, "pro"]]

np.random.seed(2)


def main():
    for p1, p1_str in player_1:
        for p2, p2_str in player_2:
            results = [0, 0]

            results_dict = []
            game_length = []
            game_time = []
            whostarts = []
            for i in range(50):
                start_temp = time.time()
                board = ataxx.Board()
                who_starts_i = np.random.randint(0, 2)
                board.turn = who_starts_i
                turn = 0
                while not board.gameover():
                    if board.turn == ataxx.BLACK:
                        d = select_depth(board, p1)
                        move = ataxx.players.alphabeta(
                            board, -999999, 999999, d, feval=ataxx.players.eval_fun, rand=True)

                    else:
                        d = select_depth(board, p2)
                        move = ataxx.players.alphabeta(
                            board,  -99999, 999999, d, feval=ataxx.players.eval_fun, rand=True)

                    turn = turn + 1

                    board.makemove(move)
                    # print("iter", i, "turno", turn)
                end_temp = time.time()
                game_time.append(end_temp - start_temp)
                game_length.append(turn)
                whostarts.append(who_starts_i)

                aux = board.result().split("-")
                results[0] += int(aux[0])
                results[1] += int(aux[1])

            aux_dict = {"p1": p1_str,
                        "p2": p2_str,
                        "results": results,
                        "starting player": whostarts,
                        "length": game_length,
                        "time": game_time}
            print(aux_dict)
            results_dict.append(aux_dict)
        joblib.dump(results_dict, "players_vs_players.job")


if __name__ == '__main__':
    main()
