import ataxx
import unittest

class TestMethods(unittest.TestCase):
    def test_perft(self):
        positions = [
            ["7/7/7/7/7/7/7 x 0 1", [1, 0, 0, 0, 0]],
            ["7/7/7/7/7/7/7 o 0 1", [1, 0, 0, 0, 0]],
            ["x5o/7/7/7/7/7/o5x x 0 1", [1, 16, 256, 6460, 155888]],
            ["x5o/7/7/7/7/7/o5x o 0 1", [1, 16, 256, 6460, 155888]],
            ["x5o/7/2-1-2/7/2-1-2/7/o5x x 0 1", [1, 14, 196, 4184, 86528]],
            ["x5o/7/2-1-2/7/2-1-2/7/o5x o 0 1", [1, 14, 196, 4184, 86528]],
            ["x5o/7/2-1-2/3-3/2-1-2/7/o5x x 0 1", [1, 14, 196, 4100]],
            ["x5o/7/2-1-2/3-3/2-1-2/7/o5x o 0 1", [1, 14, 196, 4100]],
            ["x5o/7/3-3/2-1-2/3-3/7/o5x x 0 1", [1, 16, 256, 5948]],
            ["x5o/7/3-3/2-1-2/3-3/7/o5x o 0 1", [1, 16, 256, 5948]],
            ["7/7/7/7/ooooooo/ooooooo/xxxxxxx x 0 1", [1, 1, 75, 249, 14270, 452980]],
            ["7/7/7/7/ooooooo/ooooooo/xxxxxxx o 0 1", [1, 75, 249, 14270, 452980]],
            ["7/7/7/7/xxxxxxx/xxxxxxx/ooooooo x 0 1", [1, 75, 249, 14270, 452980]],
            ["7/7/7/7/xxxxxxx/xxxxxxx/ooooooo o 0 1", [1, 1, 75, 249, 14270, 452980]],
            ["7/7/7/2x1o2/7/7/7 x 0 1", [1, 23, 419, 7887, 168317]],
            ["7/7/7/2x1o2/7/7/7 o 0 1", [1, 23, 419, 7887, 168317]],
            ["x5o/7/7/7/7/7/o5x x 100 1", [1, 0, 0, 0, 0]],
            ["x5o/7/7/7/7/7/o5x o 100 1", [1, 0, 0, 0, 0]],
            ["7/7/7/7/-------/-------/x5o x 0 1", [1, 2, 4, 13, 30, 73, 174]],
            ["7/7/7/7/-------/-------/x5o o 0 1", [1, 2, 4, 13, 30, 73, 174]],
        ]

        for fen, nodes in positions:
            board = ataxx.Board(fen)
            for idx, count in enumerate(nodes):
                self.assertTrue(board.perft(idx) == count)
