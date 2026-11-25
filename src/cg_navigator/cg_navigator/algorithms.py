"""Graph search, frontier mapping, and verification for Culling Games."""

from __future__ import annotations

import csv
import heapq
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int]
Grid = List[List[str]]

DIRECTIONS: Dict[str, Coord] = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

OPPOSITE_DIRECTION: Dict[str, str] = {
    'up': 'down',
    'down': 'up',
    'left': 'right',
    'right': 'left',
}

DELTA_TO_DIRECTION: Dict[Coord, str] = {
    delta: name for name, delta in DIRECTIONS.items()
}

SENSOR_OFFSETS: Dict[str, Coord] = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
    'up_left': (-1, -1),
    'up_right': (-1, 1),
    'down_left': (1, -1),
    'down_right': (1, 1),
}


@dataclass(frozen=True)
class SearchResult:
    """Result of a shortest-path search on an occupancy grid."""

    path: List[Coord]
    directions: List[str]
    nodes_expanded: int
    algorithm: str


def parse_flattened_grid(
    flat_grid: Sequence[str], shape: Sequence[int]
) -> Tuple[Grid, Coord, Coord]:
    """Convert GetMap flat grid to a 2D matrix and locate 'r' and 't'."""
    if len(shape) != 2:
        raise ValueError(f'Expected 2D shape, got: {shape}')
    rows, cols = int(shape[0]), int(shape[1])
    if len(flat_grid) != rows * cols:
        raise ValueError(
            f'Flattened grid length {len(flat_grid)} does not match '
            f'shape {(rows, cols)}'
        )

    grid: Grid = []
    start: Optional[Coord] = None
    target: Optional[Coord] = None

    for r in range(rows):
        row = [str(flat_grid[r * cols + c]).strip() for c in range(cols)]
        for c, cell in enumerate(row):
            if cell == 'r':
                start = (r, c)
            elif cell == 't':
                target = (r, c)
        grid.append(row)

    if start is None:
        raise ValueError("Robot cell 'r' not found in occupancy grid")
    if target is None:
        raise ValueError("Target cell 't' not found in occupancy grid")

    return grid, start, target


def manhattan_distance(a: Coord, b: Coord) -> int:
    """Admissible and consistent heuristic for 4-connected unit-cost grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_to_directions(path: Sequence[Coord]) -> List[str]:
    """Convert adjacent grid coordinates into move direction strings."""
    directions: List[str] = []
    for curr, nxt in zip(path, path[1:]):
        delta = (nxt[0] - curr[0], nxt[1] - curr[1])
        if delta not in DELTA_TO_DIRECTION:
            raise ValueError(
                f'Non-adjacent coordinates in path: {curr} -> {nxt}'
            )
        directions.append(DELTA_TO_DIRECTION[delta])
    return directions


def _reconstruct_path(
    came_from: Dict[Coord, Coord], goal: Coord
) -> List[Coord]:
    path: List[Coord] = [goal]
    curr = goal
    while curr in came_from:
        curr = came_from[curr]
        path.append(curr)
    path.reverse()
    return path


def _neighbors_4(grid: Grid, pos: Coord) -> Iterable[Coord]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    r, c = pos
    for dr, dc in DIRECTIONS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 'b':
            yield (nr, nc)


def astar(grid: Grid, start: Coord, goal: Coord) -> SearchResult:
    """Find an optimal shortest path using A* with Manhattan heuristic."""
    if start == goal:
        return SearchResult(
            path=[start], directions=[], nodes_expanded=0, algorithm='astar'
        )

    counter = 0
    h0 = manhattan_distance(start, goal)
    open_heap: List[Tuple[int, int, int, Coord]] = [(h0, h0, counter, start)]
    g_score: Dict[Coord, int] = {start: 0}
    came_from: Dict[Coord, Coord] = {}
    closed: Set[Coord] = set()
    nodes_expanded = 0

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, goal)
            return SearchResult(
                path=path,
                directions=path_to_directions(path),
                nodes_expanded=nodes_expanded,
                algorithm='astar',
            )

        tentative_g = g_score[current] + 1
        for neighbor in _neighbors_4(grid, current):
            if neighbor in closed:
                continue
            if tentative_g < g_score.get(neighbor, 10**9):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = manhattan_distance(neighbor, goal)
                counter += 1
                heapq.heappush(
                    open_heap, (tentative_g + h, h, counter, neighbor)
                )

    raise ValueError(f'No valid path found from {start} to {goal}')


def bfs(grid: Grid, start: Coord, goal: Coord) -> SearchResult:
    """Find an optimal shortest path using Breadth-First Search (BFS)."""
    if start == goal:
        return SearchResult(
            path=[start], directions=[], nodes_expanded=0, algorithm='bfs'
        )

    queue: deque[Coord] = deque([start])
    visited: Set[Coord] = {start}
    came_from: Dict[Coord, Coord] = {}
    nodes_expanded = 0

    while queue:
        current = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, goal)
            return SearchResult(
                path=path,
                directions=path_to_directions(path),
                nodes_expanded=nodes_expanded,
                algorithm='bfs',
            )

        for neighbor in _neighbors_4(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    raise ValueError(f'No valid path found from {start} to {goal}')


def solve_grid(
    grid: Grid, start: Coord, goal: Coord, algorithm: str = 'astar'
) -> SearchResult:
    """Dispatch shortest-path search using 'astar' or 'bfs'."""
    algo = algorithm.lower().strip()
    if algo == 'astar':
        return astar(grid, start, goal)
    if algo == 'bfs':
        return bfs(grid, start, goal)
    raise ValueError(
        f"Unsupported algorithm '{algorithm}'. Use 'astar' or 'bfs'."
    )


def record_sensors(
    pos: Coord,
    sensors: Dict[str, str],
    known_map: Dict[Coord, str],
) -> Optional[Coord]:
    """Record 8-connected sensor readings around pos into known_map."""
    target_found: Optional[Coord] = None
    r, c = pos
    for field_name, (dr, dc) in SENSOR_OFFSETS.items():
        val = sensors[field_name]
        nb = (r + dr, c + dc)
        if val == 't':
            known_map[nb] = 't'
            target_found = nb
        elif nb not in known_map or known_map[nb] not in ('r', 't'):
            known_map[nb] = 'f' if val == 'r' else val
    return target_found


def find_nearest_frontier(
    known_map: Dict[Coord, str],
    visited: Set[Coord],
    current_pos: Coord,
) -> Optional[List[Coord]]:
    """Find shortest path to the nearest unvisited traversable cell.

    Deliberately avoids stepping onto 't' (target) during exploration so the
    simulator does not terminate early before mapping is complete.
    """
    queue: deque[Coord] = deque([current_pos])
    seen: Set[Coord] = {current_pos}
    came_from: Dict[Coord, Coord] = {}

    while queue:
        curr = queue.popleft()
        if curr != current_pos and curr not in visited:
            return _reconstruct_path(came_from, curr)

        r, c = curr
        for dr, dc in DIRECTIONS.values():
            nb = (r + dr, c + dc)
            if nb in seen:
                continue
            cell = known_map.get(nb)
            if cell in ('f', 'r'):
                seen.add(nb)
                came_from[nb] = curr
                queue.append(nb)

    return None


def build_grid_from_map(
    known_map: Dict[Coord, str],
    start_pos: Coord,
    target_pos: Optional[Coord],
) -> Tuple[Grid, Coord, Optional[Coord], Coord]:
    """Convert a sparse relative coordinate map into a 2D occupancy grid.

    Unknown cells outside or inside unexplored regions default to 'b'.
    Returns: (grid, start_norm, target_norm, (min_r, min_c)).
    """
    if not known_map:
        raise ValueError('Cannot build grid from empty map')

    rows = [r for r, _ in known_map]
    cols = [c for _, c in known_map]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    height = max_r - min_r + 1
    width = max_c - min_c + 1
    grid: Grid = [['b' for _ in range(width)] for _ in range(height)]

    for (r, c), cell in known_map.items():
        norm_cell = 'f' if cell == 'r' else cell
        grid[r - min_r][c - min_c] = norm_cell

    start_norm = (start_pos[0] - min_r, start_pos[1] - min_c)
    grid[start_norm[0]][start_norm[1]] = 'r'

    target_norm: Optional[Coord] = None
    if target_pos is not None:
        target_norm = (target_pos[0] - min_r, target_pos[1] - min_c)
        grid[target_norm[0]][target_norm[1]] = 't'

    return grid, start_norm, target_norm, (min_r, min_c)


def simulate_directions(
    grid: Grid, start: Coord, directions: Sequence[str]
) -> Tuple[Coord, bool, Optional[int]]:
    """Simulate executing a sequence of directions on a grid.

    Returns (final_pos, valid, failed_step_index).
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    r, c = start
    for idx, direction in enumerate(directions):
        if direction not in DIRECTIONS:
            return (r, c), False, idx
        dr, dc = DIRECTIONS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] == 'b':
            return (r, c), False, idx
        r, c = nr, nc
    return (r, c), True, None


def compare_maps_and_routes(
    known_map: Dict[Coord, str],
    mapped_start_rel: Coord,
    mapped_route: Sequence[str],
    real_grid: Grid,
    real_start: Coord,
    real_target: Coord,
    real_route: Sequence[str],
) -> Dict[str, object]:
    """Audit the mapped maze and planned route against ground truth."""
    dr = real_start[0] - mapped_start_rel[0]
    dc = real_start[1] - mapped_start_rel[1]

    rows = len(real_grid)
    cols = len(real_grid[0]) if rows > 0 else 0

    matched_cells = 0
    total_mapped_cells = len(known_map)
    mismatches: List[Tuple[Coord, str, str]] = []

    for (r, c), mapped_val in known_map.items():
        rr, rc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= rc < cols:
            real_val = real_grid[rr][rc]
            norm_mapped = 'f' if mapped_val == 'r' else mapped_val
            norm_real = 'f' if real_val == 'r' else real_val
            if norm_mapped == norm_real:
                matched_cells += 1
            else:
                mismatches.append(((rr, rc), mapped_val, real_val))
        else:
            mismatches.append(((rr, rc), mapped_val, 'OUT_OF_BOUNDS'))

    final_pos, valid_moves, failed_step = simulate_directions(
        real_grid, real_start, mapped_route
    )
    reaches_target = valid_moves and (final_pos == real_target)
    same_optimal_length = len(mapped_route) == len(real_route)
    exact_same_route = list(mapped_route) == list(real_route)
    accuracy = (
        (matched_cells / total_mapped_cells * 100.0)
        if total_mapped_cells
        else 0.0
    )

    return {
        'total_mapped_cells': total_mapped_cells,
        'matched_cells': matched_cells,
        'cell_accuracy_pct': accuracy,
        'mismatches': mismatches,
        'mapped_route_length': len(mapped_route),
        'part1_route_length': len(real_route),
        'same_optimal_length': same_optimal_length,
        'exact_same_route': exact_same_route,
        'mapped_route_valid_on_real': valid_moves,
        'mapped_route_reaches_target': reaches_target,
        'failed_step': failed_step,
        'proof_passed': (
            accuracy == 100.0 and reaches_target and same_optimal_length
        ),
    }


def format_grid_ascii(
    grid: Grid, path: Optional[Sequence[Coord]] = None
) -> str:
    """Render the grid in ASCII for terminal inspection."""
    path_set = set(path) if path else set()
    lines: List[str] = []
    for r, row in enumerate(grid):
        chars: List[str] = []
        for c, cell in enumerate(row):
            if cell == 'r':
                chars.append('R')
            elif cell == 't':
                chars.append('T')
            elif (r, c) in path_set:
                chars.append('*')
            elif cell == 'b':
                chars.append('#')
            else:
                chars.append('.')
        lines.append(' '.join(chars))
    return '\n'.join(lines)


def save_grid_csv(grid: Grid, file_path: str) -> None:
    """Save a 2D occupancy grid to CSV in Culling Games format."""
    parent = Path(file_path).parent
    if parent.parts:
        parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(grid)
