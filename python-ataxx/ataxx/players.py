import ataxx
import random
import copy

# Modified from the original file from the repository https://github.com/kz04px/python-ataxx
# by Antonio Ruiz Molero


def random_move(board):
    if board.gameover():
        return ataxx.Move.null()

    moves = board.legal_moves()
    return random.choice(moves)


def greedy(board):
    if board.gameover():
        return ataxx.Move.null()

    most = -99999
    moves = []

    for move in board.legal_moves():
        board.makemove(move)
        num_black, num_white, num_gaps, num_empty = board.count()
        board.undo()

        # Maximise our advantage
        if board.turn == ataxx.BLACK:
            score = num_black - num_white
        else:
            score = num_white - num_black

        # Track most captures
        if score > most:
            most = score
            moves = []

        if score == most:
            moves.append(move)

    return random.choice(moves)

# We have modified in the function negamax and alpha beta
# the initial best_move because it returned none when
# there was no better movement and crashed because the class board cannot
# compare none with any other movement. We put ataxx.Move.null() instead to
# fix it.

# We also accept rand as argument to perform a random rearrange in the node list.
# It now accepts the alternative evaluation function via feval argument.


def negamax(board, depth, feval, root=True, rand=False):
    if depth == 0:
        black, white, _, _ = board.count()
        if board.turn == ataxx.BLACK:
            return black - white
        else:
            return white - black

    best_score = -99999
    best_move = ataxx.Move.null()

    for move in (random.sample(board.legal_moves(), len(board.legal_moves()))
                 if rand is True else board.legal_moves()):  # Rearrange of the nodes if needed
        board.makemove(move)
        score = -negamax(board, depth-1, feval, root=False)

        if score > best_score:
            best_score = score
            best_move = move

        board.undo()

    if root:
        return best_move
    else:
        return best_score


def alphabeta(board, alpha, beta, depth,  feval, root=True, rand=False):
    best_move = ataxx.Move.null()

    if depth == 0:
        return feval(board)

    for move in (random.sample(board.legal_moves(), len(board.legal_moves()))
                 if rand is True else board.legal_moves()):

        board.makemove(move)
        score = -alphabeta(board, -beta, -alpha, depth-1, feval, root=False)

        if score > alpha:

            alpha = score
            best_move = move
        if score >= beta:

            score = beta
            board.undo()
            break

        board.undo()

    if root:
        return best_move
    else:
        return alpha

# This is our implementation of minimax


def minimax(board, depth, feval,  root=True,
            maximizingPlayer=True, rand=False):
    best_move = ataxx.Move.null()
    if depth == 0:
        return feval(board)

    if maximizingPlayer:
        value = -99999
        for move in (random.sample(board.legal_moves(), len(board.legal_moves()))
                     if rand is True else board.legal_moves()):
            board.makemove(move)
            newval = minimax(board, depth-1,
                             feval, root=False, maximizingPlayer=False)
            if newval > value:
                best_move = move
                value = newval

            board.undo()

    else:
        value = 99999
        for move in (random.sample(board.legal_moves(), len(board.legal_moves()))
                     if rand is True else board.legal_moves()):
            board.makemove(move)
            newval = minimax(board, depth-1,
                             feval, root=False, maximizingPlayer=True)
            if newval < value:
                best_move = move
                value = newval

            board.undo()

    if root:
        return best_move
    else:
        return value

# Definition of the default evaluation function.


def eval_fun(board):
    black, white, _, _ = board.count()
    if board.turn == ataxx.BLACK:
        return black - white
    else:
        return white - black

# Alternative evaluation function.


def alt_val_fun(board):
    aux_board = copy.deepcopy(board)
    if board.turn == ataxx.BLACK:
        aux_board.turn = ataxx.WHITE
        return len(board.legal_moves()) - len(aux_board.legal_moves())
    else:
        aux_board.turn = ataxx.BLACK
        return len(aux_board.legal_moves()) - len(board.legal_moves())
