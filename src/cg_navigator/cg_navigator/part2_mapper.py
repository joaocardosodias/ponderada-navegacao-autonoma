"""Part 2: Sensor-based maze mapping and proof of Part 1 reproduction."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional, Set

from cg_interfaces.msg import RobotSensors
from cg_interfaces.srv import GetMap, MoveCmd, Reset

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .algorithms import (
    Coord,
    DIRECTIONS,
    OPPOSITE_DIRECTION,
    build_grid_from_map,
    compare_maps_and_routes,
    find_nearest_frontier,
    format_grid_ascii,
    parse_flattened_grid,
    path_to_directions,
    record_sensors,
    save_grid_csv,
    solve_grid,
)


def _sensors_msg_to_dict(msg: RobotSensors) -> Dict[str, str]:
    return {
        'up': msg.up.strip(),
        'down': msg.down.strip(),
        'left': msg.left.strip(),
        'right': msg.right.strip(),
        'up_left': msg.up_left.strip(),
        'up_right': msg.up_right.strip(),
        'down_left': msg.down_left.strip(),
        'down_right': msg.down_right.strip(),
    }


class Part2Mapper(Node):
    """ROS 2 node for sensor-based mapping and optimal route verification.

    1. Explores and maps the reachable maze using /culling_games/robot_sensors
       and /move_command (no /get_map calls).
    2. Reconstructs the occupancy grid and computes the optimal route with the
       same Part 1 search algorithm.
    3. Resets the current maze (/reset) and executes the planned route from
       start to target, optionally auditing against the Part 1 ground truth.
    """

    def __init__(self) -> None:
        """Initialize the Part 2 mapper node, clients, and sensor sub."""
        super().__init__('cg_part2_mapper')
        self.move_cli = self.create_client(MoveCmd, '/move_command')
        self.reset_cli = self.create_client(Reset, '/reset')
        self.get_map_cli = self.create_client(GetMap, '/get_map')

        self._latest_sensors: Optional[Dict[str, str]] = None
        self._sensor_seq: int = 0
        self.sensor_sub = self.create_subscription(
            RobotSensors,
            '/culling_games/robot_sensors',
            self._sensor_callback,
            qos_profile_sensor_data,
        )

    def wait_for_services(self) -> None:
        """Wait until /move_command and /reset services are available."""
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise KeyboardInterrupt
            self.get_logger().info('Aguardando serviço /move_command...')
        while not self.reset_cli.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise KeyboardInterrupt
            self.get_logger().info('Aguardando serviço /reset...')

    def _sensor_callback(self, msg: RobotSensors) -> None:
        self._latest_sensors = _sensors_msg_to_dict(msg)
        self._sensor_seq += 1

    def _drain_sensors(self, max_iters: int = 50) -> None:
        for _ in range(max_iters):
            prev_seq = self._sensor_seq
            rclpy.spin_once(self, timeout_sec=0.0)
            if self._sensor_seq == prev_seq:
                break

    def _spin_sleep(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.01, max(0.0, remaining)))

    def wait_for_fresh_sensors(
        self,
        min_new_msgs: int = 2,
        timeout: float = 3.0,
        expected_dir_value: Optional[tuple[str, str]] = None,
    ) -> Dict[str, str]:
        """Spin until fresh sensor readings arrive from robot_sensors.

        Waiting for >= 2 messages after a move guarantees at least one full
        10ms timer cycle elapsed after Game.py updated self.robot.pos.
        """
        self._drain_sensors()
        base_seq = self._sensor_seq
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)
            if (
                self._latest_sensors is not None
                and self._sensor_seq >= base_seq + min_new_msgs
            ):
                if expected_dir_value is None:
                    return dict(self._latest_sensors)
                exp_dir, exp_val = expected_dir_value
                if self._latest_sensors.get(exp_dir) == exp_val:
                    return dict(self._latest_sensors)

        if self._latest_sensors is None:
            raise TimeoutError(
                'Nenhuma mensagem recebida em /culling_games/robot_sensors'
            )
        raise TimeoutError(
            'Timeout aguardando leituras atualizadas em '
            '/culling_games/robot_sensors'
        )

    def send_move(
        self, direction: str, timeout_sec: float = 10.0
    ) -> MoveCmd.Response:
        """Send a single movement command to /move_command."""
        req = MoveCmd.Request()
        req.direction = direction
        future = self.move_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError(
                f'Timeout ao aguardar /move_command ({direction})'
            )
        res = future.result()
        if res is None:
            raise RuntimeError(f'Falha ao chamar /move_command ({direction})')
        return res

    def send_reset_same_map(
        self, initial_sensors: Dict[str, str], timeout_sec: float = 10.0
    ) -> Reset.Response:
        """Reset the current maze and wait for sensors to match initial."""
        req = Reset.Request()
        req.is_random = False
        req.map_name = ''
        future = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError('Timeout ao aguardar resposta de /reset')
        res = future.result()
        if res is None or not res.success:
            raise RuntimeError(
                'Falha ao chamar /reset para reiniciar o mesmo labirinto'
            )

        # Game.py aplica o reset assincronamente na thread do Pygame.
        # Aguardamos conferindo quando os sensores voltam ao estado inicial.
        deadline = time.monotonic() + 3.0
        self._spin_sleep(0.15)
        while time.monotonic() < deadline:
            try:
                sensors = self.wait_for_fresh_sensors(
                    min_new_msgs=2, timeout=0.5
                )
            except TimeoutError:
                sensors = None
            if sensors == initial_sensors:
                self._spin_sleep(0.05)
                return res
            self._spin_sleep(0.05)

        raise TimeoutError(
            'Sensores não retornaram ao estado inicial após /reset'
        )

    def fetch_ground_truth_map(
        self, timeout_sec: float = 10.0
    ) -> GetMap.Response:
        """Fetch ground-truth map from /get_map for post-mapping audit."""
        while not self.get_map_cli.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise KeyboardInterrupt
            self.get_logger().info(
                'Aguardando serviço /get_map para auditoria...'
            )
        req = GetMap.Request()
        future = self.get_map_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError('Timeout ao aguardar resposta de /get_map')
        res = future.result()
        if res is None:
            raise RuntimeError('Falha ao chamar /get_map para auditoria')
        return res

    def explore_maze(
        self, explore_delay: float = 0.0
    ) -> tuple[Dict[Coord, str], Coord, Optional[Coord], Dict[str, str], int]:
        """Map reachable cells in relative coordinates starting at (0, 0).

        Uses Nearest-Frontier Exploration via /culling_games/robot_sensors
        and /move_command.
        """
        start_rel: Coord = (0, 0)
        current_pos: Coord = start_rel
        known_map: Dict[Coord, str] = {start_rel: 'r'}
        visited: Set[Coord] = {start_rel}
        target_rel: Optional[Coord] = None
        exploration_steps = 0

        initial_sensors = self.wait_for_fresh_sensors(min_new_msgs=2)
        found = record_sensors(current_pos, initial_sensors, known_map)
        if found is not None:
            target_rel = found

        while True:
            frontier_path = find_nearest_frontier(
                known_map, visited, current_pos
            )
            if frontier_path is None:
                break

            directions = path_to_directions(frontier_path)
            for direction in directions:
                res = self.send_move(direction)
                if not res.success:
                    dr, dc = DIRECTIONS[direction]
                    blocked = (current_pos[0] + dr, current_pos[1] + dc)
                    known_map[blocked] = 'b'
                    self.get_logger().warn(
                        f'Célula {blocked} bloqueada inesperadamente; '
                        'replanejando.'
                    )
                    break

                dr, dc = DIRECTIONS[direction]
                current_pos = (current_pos[0] + dr, current_pos[1] + dc)
                exploration_steps += 1

                # Aguardar e registrar novos sensores só em células inéditas
                # (evita latência extra no backtracking).
                if current_pos not in visited:
                    visited.add(current_pos)
                    sensors = self.wait_for_fresh_sensors(
                        min_new_msgs=2,
                        expected_dir_value=(
                            OPPOSITE_DIRECTION[direction],
                            'f',
                        ),
                    )
                    found = record_sensors(current_pos, sensors, known_map)
                    if found is not None:
                        target_rel = found

                if explore_delay > 0.0:
                    self._spin_sleep(explore_delay)

        return (
            known_map,
            start_rel,
            target_rel,
            initial_sensors,
            exploration_steps,
        )

    def run(
        self,
        algorithm: str = 'astar',
        explore_delay: float = 0.0,
        replay_delay: float = 0.03,
        save_map_path: Optional[str] = None,
        verify: bool = True,
        show_ascii: bool = True,
    ) -> bool:
        """Explore the maze, reconstruct the map, reset, and replay route."""
        self.wait_for_services()
        self.get_logger().info(
            '=== PARTE 2: Mapeamento Autônomo do Labirinto ==='
        )
        self.get_logger().info(
            'Fase 1/4: Explorando labirinto usando APENAS sensores '
            '(/culling_games/robot_sensors) e /move_command (sem /get_map)...'
        )

        t_exp = time.perf_counter()
        (
            known_map,
            start_rel,
            target_rel,
            initial_sensors,
            exp_steps,
        ) = self.explore_maze(explore_delay=explore_delay)
        exp_elapsed = time.perf_counter() - t_exp

        if target_rel is None:
            self.get_logger().error(
                'Mapeamento concluído, mas o alvo (t) não foi detectado '
                'pelos sensores!'
            )
            return False

        self.get_logger().info(
            f'Mapeamento concluído em {exp_steps} movimentos '
            f'({exp_elapsed:.2f}s) | Células mapeadas: {len(known_map)} | '
            f'Alvo (relativo ao início): {target_rel}'
        )

        # Fase 2: Construir matriz 2D e rodar o algoritmo da Parte 1
        self.get_logger().info(
            f'Fase 2/4: Reconstruindo mapa e calculando rota ótima com '
            f'{algorithm.upper()} (mesmo algoritmo da Parte 1)...'
        )
        mapped_grid, start_norm, target_norm, _ = build_grid_from_map(
            known_map, start_rel, target_rel
        )
        assert target_norm is not None

        if save_map_path:
            save_grid_csv(mapped_grid, save_map_path)
            self.get_logger().info(
                f'Mapa reconstruído salvo em: {save_map_path}'
            )

        mapped_result = solve_grid(
            mapped_grid, start_norm, target_norm, algorithm=algorithm
        )
        self.get_logger().info(
            f'Mapa reconstruído ({len(mapped_grid)}x'
            f'{len(mapped_grid[0])}) | Rota ótima calculada a partir do '
            f'mapa mapeado: {len(mapped_result.directions)} passos '
            f'(nós expandidos: {mapped_result.nodes_expanded})'
        )

        if show_ascii:
            self.get_logger().info(
                'Mapa reconstruído com rota ótima (*):\n'
                + format_grid_ascii(mapped_grid, mapped_result.path)
            )

        # Fase 3: Reiniciar o mesmo labirinto para comprovar a reprodução
        self.get_logger().info(
            'Fase 3/4: Reiniciando o mesmo labirinto '
            '(/reset is_random=false) para comprovação...'
        )
        reset_res = self.send_reset_same_map(initial_sensors)
        self.get_logger().info(
            f'Labirinto reiniciado: {reset_res.loaded_map_name}'
        )

        # Auditoria opcional contra a Parte 1 (rota mapped_result congelada)
        proof_ok = True
        if verify:
            gt_res = self.fetch_ground_truth_map()
            real_grid, real_start, real_target = parse_flattened_grid(
                gt_res.occupancy_grid_flattened,
                gt_res.occupancy_grid_shape,
            )
            part1_result = solve_grid(
                real_grid, real_start, real_target, algorithm=algorithm
            )
            audit = compare_maps_and_routes(
                known_map=known_map,
                mapped_start_rel=start_rel,
                mapped_route=mapped_result.directions,
                real_grid=real_grid,
                real_start=real_start,
                real_target=real_target,
                real_route=part1_result.directions,
            )
            proof_ok = bool(audit['proof_passed'])
            matched = audit['matched_cells']
            total = audit['total_mapped_cells']
            pct = audit['cell_accuracy_pct']
            p1_len = audit['part1_route_length']
            p2_len = audit['mapped_route_length']
            same_opt = audit['same_optimal_length']
            same_seq = audit['exact_same_route']
            reaches = audit['mapped_route_reaches_target']
            self.get_logger().info(
                '--- COMPROVAÇÃO (Mapa Mapeado vs Gabarito Parte 1) ---\n'
                f'  Acurácia das células mapeadas : {matched}/{total} '
                f'({pct:.2f}%)\n'
                f'  Rota Parte 1 (com /get_map)   : {p1_len} passos\n'
                f'  Rota Parte 2 (mapa mapeado)   : {p2_len} passos\n'
                f'  Mesmo custo ótimo             : {same_opt}\n'
                f'  Sequência de passos idêntica  : {same_seq}\n'
                f'  Rota válida no mapa real      : {reaches}\n'
                f'  Comprovação aprovada          : {proof_ok}'
            )

        # Fase 4: Executar a rota planejada pelo mapa construído
        self.get_logger().info(
            f'Fase 4/4: Executando rota ótima '
            f'({len(mapped_result.directions)} passos) gerada exclusivamente '
            'a partir do mapa mapeado...'
        )
        last_res: Optional[MoveCmd.Response] = None
        for step_idx, direction in enumerate(
            mapped_result.directions, start=1
        ):
            last_res = self.send_move(direction)
            if not last_res.success:
                pos = [int(v) for v in last_res.robot_pos]
                self.get_logger().error(
                    f'Movimento falhou no passo '
                    f'{step_idx}/{len(mapped_result.directions)} '
                    f'({direction}). Posição: {pos}'
                )
                return False
            if replay_delay > 0.0:
                self._spin_sleep(replay_delay)

        if start_norm == target_norm:
            reached = True
        else:
            reached = (
                last_res is not None
                and tuple(last_res.robot_pos) == tuple(last_res.target_pos)
            )

        if reached and proof_ok:
            self.get_logger().info(
                f'SUCESSO! Alvo alcançado em '
                f'{len(mapped_result.directions)} passos reproduzindo a '
                'rota ótima da Parte 1 usando o mapa criado pelo robô!'
            )
            return True

        self.get_logger().error(
            f'Falha na comprovação (reached={reached}, proof_ok={proof_ok}).'
        )
        return False


def main(args=None) -> None:
    """CLI entrypoint for the Part 2 autonomous mapper node."""
    parser = argparse.ArgumentParser(
        description='Parte 2 — Mapeamento autônomo do labirinto e '
        'reprodução da rota ótima da Parte 1.'
    )
    parser.add_argument(
        '--algorithm',
        choices=['astar', 'bfs'],
        default='astar',
        help='Algoritmo de busca usado sobre o mapa reconstruído '
        '(padrão: astar).',
    )
    parser.add_argument(
        '--explore-delay',
        type=float,
        default=0.0,
        help='Atraso em segundos entre movimentos durante a exploração '
        '(padrão: 0.0).',
    )
    parser.add_argument(
        '--replay-delay',
        type=float,
        default=0.03,
        help='Atraso em segundos entre movimentos na execução da rota ótima '
        '(padrão: 0.03).',
    )
    parser.add_argument(
        '--save-map',
        type=str,
        default=None,
        help='Caminho opcional de CSV para salvar o mapa reconstruído.',
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Desativa a auditoria pós-planejamento contra /get_map.',
    )
    parser.add_argument(
        '--no-ascii',
        action='store_true',
        help='Não imprime o mapa ASCII no terminal.',
    )
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = Part2Mapper()
    success = False
    try:
        success = node.run(
            algorithm=parsed.algorithm,
            explore_delay=parsed.explore_delay,
            replay_delay=parsed.replay_delay,
            save_map_path=parsed.save_map,
            verify=not parsed.no_verify,
            show_ascii=not parsed.no_ascii,
        )
    except KeyboardInterrupt:
        node.get_logger().info('Interrompido pelo usuário.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
