"""
SGF Parser — parse SGF game records and replay board states.
Supports 9x9, 13x13, 19x19 boards with capture logic.
"""

import re
from typing import List, Tuple, Optional

BLACK = 1
WHITE = 2
EMPTY = 0


class BoardState:
    """Immutable-ish board state with stone placement + capture logic."""

    def __init__(self, size: int = 19):
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]

    def copy(self) -> 'BoardState':
        b = BoardState(self.size)
        b.grid = [row[:] for row in self.grid]
        return b

    def place(self, row: int, col: int, color: int) -> 'BoardState':
        """Place a stone and handle captures. Returns new BoardState."""
        new = self.copy()
        new.grid[row][col] = color
        opponent = WHITE if color == BLACK else BLACK
        # Check captures for opponent groups adjacent to the placed stone
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if new.grid[nr][nc] == opponent:
                    group = new._find_group(nr, nc)
                    if not new._has_liberty(group):
                        for gr, gc in group:
                            new.grid[gr][gc] = EMPTY
        # Check self-capture (suicide) — remove own group if no liberties
        own_group = new._find_group(row, col)
        if not new._has_liberty(own_group):
            for gr, gc in own_group:
                new.grid[gr][gc] = EMPTY
        return new

    def _find_group(self, row: int, col: int) -> set:
        """Flood-fill to find connected group of same color."""
        color = self.grid[row][col]
        if color == EMPTY:
            return set()
        visited = set()
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if r < 0 or r >= self.size or c < 0 or c >= self.size:
                continue
            if self.grid[r][c] != color:
                continue
            visited.add((r, c))
            stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        return visited

    def _has_liberty(self, group: set) -> bool:
        """Check if a group has at least one liberty."""
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.grid[nr][nc] == EMPTY:
                        return True
        return False


def parse_sgf(sgf_text: str) -> dict:
    """
    Parse an SGF file and extract game info + move list.
    Returns dict with 'size', 'moves' [(color, row, col), ...],
    and 'setup_black' / 'setup_white' for AB/AW properties.
    """
    # Extract board size
    sz_match = re.search(r'SZ\[(\d+)\]', sgf_text)
    size = int(sz_match.group(1)) if sz_match else 19

    # Extract setup stones (AB/AW)
    setup_black = []
    setup_white = []
    for m in re.finditer(r'AB((?:\[[a-z]{2}\])+)', sgf_text):
        coords = re.findall(r'\[([a-z]{2})\]', m.group(0))
        for c in coords:
            col, row = ord(c[0]) - ord('a'), ord(c[1]) - ord('a')
            if 0 <= row < size and 0 <= col < size:
                setup_black.append((row, col))

    for m in re.finditer(r'AW((?:\[[a-z]{2}\])+)', sgf_text):
        coords = re.findall(r'\[([a-z]{2})\]', m.group(0))
        for c in coords:
            col, row = ord(c[0]) - ord('a'), ord(c[1]) - ord('a')
            if 0 <= row < size and 0 <= col < size:
                setup_white.append((row, col))

    # Extract moves (B[xx] and W[xx])
    # Use lookbehind to avoid matching AB[xx]/AW[xx] setup properties
    moves = []
    nodes = re.split(r';', sgf_text)
    for node in nodes:
        # Black move — must not be preceded by uppercase (e.g. AB)
        bm = re.search(r'(?<![A-Z])B\[([a-z]{2})\]', node)
        if bm:
            c = bm.group(1)
            col, row = ord(c[0]) - ord('a'), ord(c[1]) - ord('a')
            if 0 <= row < size and 0 <= col < size:
                moves.append((BLACK, row, col))

        # White move — must not be preceded by uppercase (e.g. AW)
        wm = re.search(r'(?<![A-Z])W\[([a-z]{2})\]', node)
        if wm:
            c = wm.group(1)
            col, row = ord(c[0]) - ord('a'), ord(c[1]) - ord('a')
            if 0 <= row < size and 0 <= col < size:
                moves.append((WHITE, row, col))

    return {
        'size': size,
        'moves': moves,
        'setup_black': setup_black,
        'setup_white': setup_white,
    }


def replay_game(sgf_text: str, sample_every: int = 10) -> List[BoardState]:
    """
    Parse an SGF and replay, returning board states sampled every N moves.
    Always includes the final state.
    """
    game = parse_sgf(sgf_text)
    size = game['size']

    board = BoardState(size)

    # Apply setup stones
    for r, c in game['setup_black']:
        board.grid[r][c] = BLACK
    for r, c in game['setup_white']:
        board.grid[r][c] = WHITE

    states = []
    for i, (color, row, col) in enumerate(game['moves']):
        board = board.place(row, col, color)
        if (i + 1) % sample_every == 0:
            states.append(board.copy())

    # Always include final state
    if game['moves'] and (len(game['moves']) % sample_every != 0):
        states.append(board.copy())

    # Include initial state if no moves
    if not states:
        states.append(board.copy())

    return states


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sgf_parser.py <sgf_file>")
        sys.exit(1)
    with open(sys.argv[1], 'r', encoding='utf-8', errors='ignore') as f:
        sgf = f.read()
    game = parse_sgf(sgf)
    print(f"Board size: {game['size']}")
    print(f"Setup: B={len(game['setup_black'])}, W={len(game['setup_white'])}")
    print(f"Moves: {len(game['moves'])}")
    states = replay_game(sgf, sample_every=50)
    print(f"Sampled {len(states)} board states")
    for s in states:
        b = sum(1 for r in s.grid for c in r if c == BLACK)
        w = sum(1 for r in s.grid for c in r if c == WHITE)
        print(f"  B={b}, W={w}")
