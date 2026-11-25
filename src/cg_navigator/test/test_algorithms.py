"""Unit and offline simulation tests for Part 1 and Part 2 algorithms."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict, Optional, Set, Tuple
from unittest.mock import MagicMock

from cg_navigator.algorithms import (
    Coord,
    DIRECTIONS,
    Grid,
    SENSOR_OFFSETS,
    astar,
    bfs,
    build_grid_from_map,
    compare_maps_and_routes,
    find_nearest_frontier,
    parse_flattened_grid,
    path_to_directions,
    record_sensors,
    save_grid_csv,
    simulate_directions,
    solve_grid,
)

import pytest

MAPS_DIR = Path(__file__).resolve().parents[2] / 'cg' / 'maps'


def _ensure_cg_interfaces_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub cg_interfaces in sys.modules when install/ is not sourced."""
    if importlib.util.find_spec('cg_interfaces') is not None:
        return
    pkg = ModuleType('cg_interfaces')
    msg_mod = ModuleType('cg_interfaces.msg')
    srv_mod = ModuleType('cg_interfaces.srv')
    msg_mod.RobotSensors = MagicMock
    srv_mod.GetMap = MagicMock
    srv_mod.MoveCmd = MagicMock
    srv_mod.Reset = MagicMock
    pkg.msg = msg_mod
    pkg.srv = srv_mod
    monkeypatch.setitem(sys.modules, 'cg_interfaces', pkg)
    monkeypatch.setitem(sys.modules, 'cg_interfaces.msg', msg_mod)
    monkeypatch.setitem(sys.modules, 'cg_interfaces.srv', srv_mod)


def _load_csv_map(csv_path: Path) -> Tuple[Grid, Coord, Coord]:
    with open(csv_path, 'r') as f:
        rows = [row for row in csv.reader(f) if row]
    flat = [cell for row in rows for cell in row]
    return parse_flattened_grid(flat, (len(rows), len(rows[0])))


def _simulate_part2_offline(
    real_grid: Grid, real_start: Coord
) -> Tuple[Dict[Coord, str], Coord, Optional[Coord], int]:
    """Run Part 2 frontier mapping loop against an in-memory simulator."""
    sim_grid = [row[:] for row in real_grid]
    sim_pos = real_start

    def read_sensors(pos: Coord) -> Dict[str, str]:
        r, c = pos
        return {
            name: sim_grid[r + dr][c + dc]
            for name, (dr, dc) in SENSOR_OFFSETS.items()
        }

    def move_sim(direction: str) -> bool:
        nonlocal sim_pos
        dr, dc = DIRECTIONS[direction]
        nr, nc = sim_pos[0] + dr, sim_pos[1] + dc
        if sim_grid[nr][nc] == 'b':
            return False
        sim_grid[sim_pos[0]][sim_pos[1]] = 'f'
        sim_grid[nr][nc] = 'r'
        sim_pos = (nr, nc)
        return True

    start_rel: Coord = (0, 0)
    curr_rel: Coord = start_rel
    known_map: Dict[Coord, str] = {start_rel: 'r'}
    visited: Set[Coord] = {start_rel}
    target_rel: Optional[Coord] = None
    steps = 0

    found = record_sensors(curr_rel, read_sensors(sim_pos), known_map)
    if found is not None:
        target_rel = found

    while True:
        frontier_path = find_nearest_frontier(known_map, visited, curr_rel)
        if frontier_path is None:
            break
        for direction in path_to_directions(frontier_path):
            assert move_sim(direction), (
                f'Unexpected collision moving {direction}'
            )
            dr, dc = DIRECTIONS[direction]
            curr_rel = (curr_rel[0] + dr, curr_rel[1] + dc)
            steps += 1
            visited.add(curr_rel)
            found = record_sensors(curr_rel, read_sensors(sim_pos), known_map)
            if found is not None:
                target_rel = found

    return known_map, start_rel, target_rel, steps


def test_parse_flattened_grid_and_basic_search() -> None:
    """Verify grid parsing and shortest-path agreement on a small maze."""
    raw = [
        ['b', 'b', 'b', 'b', 'b'],
        ['b', 'r', 'f', 'f', 'b'],
        ['b', 'b', 'b', 'f', 'b'],
        ['b', 't', 'f', 'f', 'b'],
        ['b', 'b', 'b', 'b', 'b'],
    ]
    flat = [c for row in raw for c in row]
    grid, start, target = parse_flattened_grid(flat, (5, 5))
    assert start == (1, 1)
    assert target == (3, 1)

    res_astar = astar(grid, start, target)
    res_bfs = bfs(grid, start, target)
    assert res_astar.directions == [
        'right', 'right', 'down', 'down', 'left', 'left'
    ]
    assert len(res_astar.directions) == len(res_bfs.directions) == 6


@pytest.mark.parametrize('csv_path', sorted(MAPS_DIR.glob('*.csv')))
def test_part1_astar_matches_bfs_optimality_on_all_maps(
    csv_path: Path,
) -> None:
    """Ensure A* matches BFS path length across all benchmark maps."""
    grid, start, target = _load_csv_map(csv_path)

    res_astar = astar(grid, start, target)
    res_bfs = bfs(grid, start, target)

    assert len(res_astar.directions) == len(res_bfs.directions)
    assert res_astar.nodes_expanded <= res_bfs.nodes_expanded

    final_pos, valid, failed = simulate_directions(
        grid, start, res_astar.directions
    )
    assert valid and failed is None
    assert final_pos == target


@pytest.mark.parametrize('csv_path', sorted(MAPS_DIR.glob('*.csv')))
def test_part2_mapping_reproduces_part1_on_all_maps(csv_path: Path) -> None:
    """Ensure Part 2 mapping reproduces Part 1 optimal routes on all maps."""
    real_grid, real_start, real_target = _load_csv_map(csv_path)
    part1_result = astar(real_grid, real_start, real_target)

    known_map, start_rel, target_rel, _ = _simulate_part2_offline(
        real_grid, real_start
    )
    assert target_rel is not None

    mapped_grid, start_norm, target_norm, _ = build_grid_from_map(
        known_map, start_rel, target_rel
    )
    assert target_norm is not None

    mapped_result = astar(mapped_grid, start_norm, target_norm)
    audit = compare_maps_and_routes(
        known_map=known_map,
        mapped_start_rel=start_rel,
        mapped_route=mapped_result.directions,
        real_grid=real_grid,
        real_start=real_start,
        real_target=real_target,
        real_route=part1_result.directions,
    )

    assert audit['cell_accuracy_pct'] == 100.0
    assert audit['same_optimal_length'] is True
    assert audit['mapped_route_reaches_target'] is True
    assert audit['proof_passed'] is True


def test_part2_when_target_is_chokepoint() -> None:
    """Avoid stepping onto 't' while still finding the optimal route."""
    grid = [
        ['b', 'b', 'b', 'b', 'b', 'b', 'b'],
        ['b', 'r', 'f', 't', 'f', 'f', 'b'],
        ['b', 'b', 'b', 'b', 'b', 'b', 'b'],
    ]
    real_start = (1, 1)
    real_target = (1, 3)

    part1_result = astar(grid, real_start, real_target)
    known_map, start_rel, target_rel, _ = _simulate_part2_offline(
        grid, real_start
    )
    assert target_rel == (0, 2)

    mapped_grid, start_norm, target_norm, _ = build_grid_from_map(
        known_map, start_rel, target_rel
    )
    assert target_norm is not None
    mapped_result = astar(mapped_grid, start_norm, target_norm)

    assert mapped_result.directions == part1_result.directions == [
        'right', 'right'
    ]


def test_parse_flattened_grid_errors() -> None:
    """Validate error handling for invalid grid shapes and missing cells."""
    with pytest.raises(ValueError, match='2D shape'):
        parse_flattened_grid(['r', 't'], (2,))
    with pytest.raises(ValueError, match='does not match shape'):
        parse_flattened_grid(['r', 't'], (2, 2))
    with pytest.raises(ValueError, match="Robot cell 'r'"):
        parse_flattened_grid(['f', 't', 'f', 'b'], (2, 2))
    with pytest.raises(ValueError, match="Target cell 't'"):
        parse_flattened_grid(['r', 'f', 'f', 'b'], (2, 2))


def test_search_and_path_error_paths() -> None:
    """Validate error handling for solve_grid, path_to_directions, build."""
    grid = [
        ['b', 'b', 'b', 'b'],
        ['b', 'r', 'b', 't'],
        ['b', 'b', 'b', 'b'],
    ]
    with pytest.raises(ValueError, match='Unsupported algorithm'):
        solve_grid(grid, (1, 1), (1, 3), algorithm='dijkstra')
    with pytest.raises(ValueError, match='No valid path'):
        solve_grid(grid, (1, 1), (1, 3), algorithm='astar')
    with pytest.raises(ValueError, match='No valid path'):
        solve_grid(grid, (1, 1), (1, 3), algorithm='bfs')
    with pytest.raises(ValueError, match='Non-adjacent'):
        path_to_directions([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match='empty map'):
        build_grid_from_map({}, (0, 0), None)


def test_simulate_directions_error_paths() -> None:
    """Verify invalid direction, wall collision, and out-of-bounds checks."""
    grid = [
        ['r', 'f'],
        ['b', 't'],
    ]
    assert simulate_directions(grid, (0, 0), ['jump']) == ((0, 0), False, 0)
    assert simulate_directions(grid, (0, 0), ['down']) == ((0, 0), False, 0)
    assert simulate_directions(grid, (0, 0), ['up']) == ((0, 0), False, 0)
    assert simulate_directions(grid, (0, 0), ['right', 'down']) == (
        (1, 1), True, None
    )


def test_zero_step_search_astar_and_bfs() -> None:
    """Verify start == goal returns 0-step result for both A* and BFS."""
    grid = [['r']]
    for algo in ('astar', 'bfs'):
        res = solve_grid(grid, (0, 0), (0, 0), algorithm=algo)
        assert res.path == [(0, 0)]
        assert res.directions == []
        assert res.nodes_expanded == 0
        assert res.algorithm == algo


def test_save_grid_csv_creates_nested_parents(tmp_path: Path) -> None:
    """Verify save_grid_csv creates missing parent directories."""
    out_file = tmp_path / 'nested' / 'dir' / 'map.csv'
    grid = [['b', 'r'], ['f', 't']]
    save_grid_csv(grid, str(out_file))
    assert out_file.is_file()
    with open(out_file, 'r') as f:
        assert list(csv.reader(f)) == grid


def test_compare_maps_and_routes_mismatches_and_oob() -> None:
    """Verify compare_maps_and_routes reports cell mismatches and OOB."""
    real_grid = [
        ['b', 'b', 'b'],
        ['b', 'r', 't'],
        ['b', 'b', 'b'],
    ]
    known_map = {
        (0, 0): 'r',
        (0, 1): 'b',  # mismatch vs 't'
        (-5, -5): 'f',  # out of bounds
    }
    audit = compare_maps_and_routes(
        known_map=known_map,
        mapped_start_rel=(0, 0),
        mapped_route=['right'],
        real_grid=real_grid,
        real_start=(1, 1),
        real_target=(1, 2),
        real_route=['right'],
    )
    assert audit['matched_cells'] == 1
    assert audit['total_mapped_cells'] == 3
    assert audit['proof_passed'] is False
    assert ((1, 2), 'b', 't') in audit['mismatches']
    assert ((-4, -4), 'f', 'OUT_OF_BOUNDS') in audit['mismatches']


def test_part2_wait_for_fresh_sensors_raises_timeout_not_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure wait_for_fresh_sensors raises TimeoutError on stale sensors."""
    _ensure_cg_interfaces_stub(monkeypatch)
    from cg_navigator import part2_mapper

    node = part2_mapper.Part2Mapper.__new__(part2_mapper.Part2Mapper)
    node._latest_sensors = dict.fromkeys(SENSOR_OFFSETS, 'f')
    node._sensor_seq = 5

    monkeypatch.setattr(part2_mapper.rclpy, 'spin_once', lambda *a, **k: None)

    with pytest.raises(TimeoutError):
        node.wait_for_fresh_sensors(min_new_msgs=2, timeout=0.02)

    def spin_wrong_dir(*args, **kwargs) -> None:
        if kwargs.get('timeout_sec', 0.0) > 0.0:
            node._latest_sensors = dict.fromkeys(SENSOR_OFFSETS, 'b')
            node._sensor_seq += 1

    monkeypatch.setattr(part2_mapper.rclpy, 'spin_once', spin_wrong_dir)
    with pytest.raises(TimeoutError):
        node.wait_for_fresh_sensors(
            min_new_msgs=1,
            timeout=0.02,
            expected_dir_value=('down', 'f'),
        )


def test_part1_and_part2_run_zero_step_start_equals_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure Part1Navigator.run and Part2Mapper.run succeed for 0 steps."""
    _ensure_cg_interfaces_stub(monkeypatch)
    from cg_navigator import part1_navigator, part2_mapper

    logger = MagicMock()

    p1 = part1_navigator.Part1Navigator.__new__(
        part1_navigator.Part1Navigator
    )
    p1.get_logger = lambda: logger
    p1.wait_for_services = lambda: None
    p1.fetch_map = lambda: SimpleNamespace(
        occupancy_grid_flattened=['r'],
        occupancy_grid_shape=[1, 1],
    )
    p1.send_move = MagicMock(
        side_effect=AssertionError('send_move should not be called')
    )
    monkeypatch.setattr(
        part1_navigator,
        'parse_flattened_grid',
        lambda flat, shape: ([['r']], (0, 0), (0, 0)),
    )
    assert p1.run(delay=0.0, show_ascii=False) is True

    p2 = part2_mapper.Part2Mapper.__new__(part2_mapper.Part2Mapper)
    p2.get_logger = lambda: logger
    p2.wait_for_services = lambda: None
    p2.explore_maze = lambda explore_delay=0.0: (
        {(0, 0): 'r'},
        (0, 0),
        (0, 0),
        dict.fromkeys(SENSOR_OFFSETS, 'b'),
        0,
    )
    p2.send_reset_same_map = lambda init_s: SimpleNamespace(
        loaded_map_name='zero_step.csv'
    )
    p2.send_move = MagicMock(
        side_effect=AssertionError('send_move should not be called')
    )
    assert p2.run(replay_delay=0.0, verify=False, show_ascii=False) is True
