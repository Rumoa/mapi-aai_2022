"""
Game of Reversi
"""

# Antonio Ruiz Molero.
# Modified from the original file from openai gym from the repository
# https://github.com/pigooosuke/gym_reversi

# Some comments are from the original file while others are related to our modification
# The file includes a class ReversiEnv which contains all the method that openai gym requires

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding

# Make random policy


def make_random_policy(np_random):
    def random_policy(state, player_color):
        possible_places = ReversiEnv.get_possible_actions(state, player_color)
        # No places left
        if len(possible_places) == 0:
            d = state.shape[-1]
            return d**2 + 1
        a = np_random.randint(len(possible_places))
        return possible_places[a]
    return random_policy

# We  include a manual policy where it asks for the selection of the movement.


def make_user_policy():
    def ask_policy(state, player_color):

        possible_places = ReversiEnv.get_possible_actions(state, player_color)
        # No places left
        if len(possible_places) == 0:
            d = state.shape[-1]
            return d**2 + 1
        print(f"You are playing as {player_color} (B=0,W=1)")
        print("Possible movs")
        print(possible_places)
        x = input("Choose first coord\n")
        y = input("Chose second coord\n")
        choice = [int(x)-1, int(y)-1]
        print(choice)
        return ReversiEnv.coordinate_to_action(state, choice)
    return ask_policy


class ReversiEnv(gym.Env):
    """
    Although is called reversi, it has been modified to mimic Ataxx.
    Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, player_color, opponent, observation_type, illegal_place_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_place_mode: What to do when the agent makes an illegal place. Choices: 'raise' or 'lose'
            board_size: size of the Reversi board
        """
        assert isinstance(
            board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size
        self.render_in_between = False
        colormap = {
            'black': ReversiEnv.BLACK,
            'white': ReversiEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error(
                "player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        # MOD* , we have included raise and pass
        assert illegal_place_mode in ['lose', 'raise', 'pass']
        self.illegal_place_mode = illegal_place_mode

        if self.observation_type != 'numpy3c':
            raise error.Error(
                'Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign and pass
        self.action_space = spaces.Discrete(self.board_size ** 2 + 2)
        observation = self.reset()
        # Definition of observation space as defined in the report.
        self.observation_space = spaces.Box(np.zeros(
            observation.shape, dtype=np.float32), np.ones(observation.shape, dtype=np.float32))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            elif self.opponent == "manual":
                self.opponent_policy = make_user_policy()
                self.render_in_between is True
            else:
                raise error.Error(
                    'Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    # We include this function to change the policy. This allows to initialize
    # the object and use stable baselines 3 and later change the opponent to manual so
    # we can play against the machine.
    def change_policy(self, opponent):
        if isinstance(opponent, str):
            if opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            elif opponent == "manual":
                self.opponent_policy = make_user_policy()
                self.render_in_between = True
            else:
                raise error.Error(
                    'Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

    def reset(self):
        # init board setting
        self.state = np.zeros(
            (3, self.board_size, self.board_size), dtype=np.float32)

        # Initalization of the initial state. We have modified it the initial places are the one
        # in ataxx.
        self.state[2, :, :] = 1.0
        self.state[2, 0, 0] = 0
        self.state[2, 0, self.board_size-1] = 0
        self.state[2, self.board_size-1, 0] = 0
        self.state[2, self.board_size-1, self.board_size-1] = 0

        self.state[1, 0, 0] = 1
        self.state[0, 0, self.board_size-1] = 1
        self.state[0, self.board_size-1, 0] = 1
        self.state[1, self.board_size-1, self.board_size-1] = 1

        # The first to play is black
        self.to_play = ReversiEnv.BLACK
        self.possible_actions = ReversiEnv.get_possible_actions(
            self.state, self.to_play)
        # set the game as unfinished.
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state)
            ReversiEnv.make_place(self.state, a, ReversiEnv.BLACK)
            self.to_play = ReversiEnv.WHITE
        return self.state
    # Perform an action

    def step(self, action):
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}
        # pass movement
        if ReversiEnv.pass_place(self.board_size, action):
            pass
        # Resign action
        elif ReversiEnv.resign_place(self.board_size, action):
            return self.state, -1, True, {'state': self.state}
        # If the selected action is not valid, raise lose or pass
        # depending on configuration.
        elif not ReversiEnv.valid_place(self.state, action, self.player_color):
            if self.illegal_place_mode == 'raise':
                raise
            elif self.illegal_place_mode == 'lose':
                # Automatic loss on illegal place
                self.done = True
                return self.state, -1., True, {'state': self.state}
            elif self.illegal_place_mode == 'pass':
                pass
            else:
                raise error.Error('Unsupported illegal place action: {}'.format(
                    self.illegal_place_mode))
        else:
            # Execute the action in the board
            ReversiEnv.make_place(self.state, action, self.player_color)

        # the attribute render in between is used in the manual mode to
        # show the movement of the opponent
        if self.render_in_between is True:
            self.render()
        # Opponent play
        a = self.opponent_policy(self.state, 1 - self.player_color)

        # Making place if there are places left
        if a is not None:
            if ReversiEnv.pass_place(self.board_size, a):
                pass
            elif ReversiEnv.resign_place(self.board_size, a):
                return self.state, 1, True, {'state': self.state}
            elif not ReversiEnv.valid_place(self.state, a, 1 - self.player_color):
                if self.illegal_place_mode == 'raise':
                    raise
                elif self.illegal_place_mode == 'lose':
                    # Automatic loss on illegal place
                    self.done = True
                    return self.state, 1., True, {'state': self.state}
                else:
                    raise error.Error('Unsupported illegal place action: {}'.format(
                        self.illegal_place_mode))
            else:
                ReversiEnv.make_place(self.state, a, 1 - self.player_color)

        self.possible_actions = ReversiEnv.get_possible_actions(
            self.state, self.player_color)
        # The reward is 0 if the game is not finished, and -1 or 1 when the game is finished, depending on the player.
        reward = ReversiEnv.game_finished(self.state)
        if self.player_color == ReversiEnv.WHITE:
            reward = - reward
        self.done = reward != 0
        return self.state, reward, self.done, {'state': self.state}

    # Render function.
    def render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(' ' * 7)
        for j in range(board.shape[1]):
            outfile.write(' ' + str(j + 1) + '  | ')
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write('-' * (board.shape[1] * 6 - 1))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(' ' + str(i + 1) + '  |')
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write('  O  ')
                elif board[0, i, j] == 1:
                    outfile.write('  B  ')
                else:
                    outfile.write('  W  ')
                outfile.write('|')
            outfile.write('\n')
            outfile.write(' ')
            outfile.write('-' * (board.shape[1] * 7 - 1))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    @staticmethod
    def resign_place(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def pass_place(board_size, action):
        return action == board_size ** 2 + 1

    # This function was rewritten from scratch to include the ataxx  rules
    @staticmethod
    def get_possible_actions(board, player_color):
        actions = []
        d = board.shape[-1]
        opponent_color = 1 - player_color
        for pos_x in range(d):  # Moves along the board
            for pos_y in range(d):
                if (board[player_color, pos_x, pos_y] != 1):  # If the board is empty, continue
                    continue

                for dx in [-2, -1, 0, 1, 2]:  # dx to look left and right two squares far
                    for dy in [-2, -1, 0, 1, 2]:  # dy to look up and down two squares far

                        if(dx == 0 and dy == 0):  # if dx and dy = 0 , pass
                            continue
                        nx = pos_x + dx  # new  x coordinate
                        ny = pos_y + dy  # new y coordinate

                        # pass if we look outside the board
                        if (nx not in range(d) or ny not in range(d)):
                            continue
                        # If the place we are lookin is free and we start lookin from a piece of our color, add the action.
                        if (board[2, nx, ny] == 1) and (board[player_color, pos_x, pos_y] == 1) and (board[opponent_color, pos_x, pos_y] == 0):
                            actions.append(nx * d + ny)
        # If there is no actions available, add another one as pass.
        if len(actions) == 0:
            actions = [d**2 + 1]
        return actions

    # Check if an action is valid
    @staticmethod
    def valid_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)
        # check whether there is any empty places
        if board[2, coords[0], coords[1]] == 1:
            # Get valid movements.
            valid_movs = ReversiEnv.get_possible_actions(board, player_color)
            if action in valid_movs:
                return True
        else:
            return False
    # Change the board state. This function has been modified to convert all opponent pieces
    # surrounding the new one.

    @staticmethod
    def make_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)

        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]

        board[2, pos_x, pos_y] = 0
        board[player_color, pos_x, pos_y] = 1
        board[opponent_color, pos_x, pos_y] = 0

        for dx in [-1, 0, 1]:  # check around the new position
            for dy in [-1, 0, 1]:

                if(dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy

                if (nx not in range(d) or ny not in range(d)):
                    continue

                # change the opponent  pieces
                if (board[opponent_color, nx, ny] == 1):
                    board[2, nx, ny] = 0
                    board[player_color, nx, ny] = 1
                    board[opponent_color, nx, ny] = 0

        return board

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    @staticmethod
    def game_finished(board):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        d = board.shape[-1]

        player_score_x, player_score_y = np.where(board[0, :, :] == 1)
        player_score = len(player_score_x)
        opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
        opponent_score = len(opponent_score_x)
        if player_score == 0:
            return -1
        elif opponent_score == 0:
            return 1
        else:
            free_x, free_y = np.where(board[2, :, :] == 1)
            if free_x.size == 0:
                if player_score > (d**2)/2:
                    return 1
                elif player_score == (d**2)/2:
                    return 1
                else:
                    return -1
            else:
                return 0
