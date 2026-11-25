"""Part 1: Autonomous maze navigation with known map (/get_map + A*/BFS)."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from cg_interfaces.srv import GetMap, MoveCmd

import rclpy
from rclpy.node import Node

from .algorithms import format_grid_ascii, parse_flattened_grid, solve_grid


class Part1Navigator(Node):
    """ROS 2 node that fetches the map and navigates via the optimal route."""

    def __init__(self) -> None:
        """Initialize the Part 1 navigator node and service clients."""
        super().__init__('cg_part1_navigator')
        self.get_map_cli = self.create_client(GetMap, '/get_map')
        self.move_cli = self.create_client(MoveCmd, '/move_command')

    def wait_for_services(self) -> None:
        """Wait until /get_map and /move_command services are available."""
        while not self.get_map_cli.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise KeyboardInterrupt
            self.get_logger().info('Aguardando serviço /get_map...')
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise KeyboardInterrupt
            self.get_logger().info('Aguardando serviço /move_command...')

    def fetch_map(self, timeout_sec: float = 10.0) -> GetMap.Response:
        """Request the full occupancy grid from /get_map."""
        req = GetMap.Request()
        future = self.get_map_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError('Timeout ao aguardar resposta de /get_map')
        res = future.result()
        if res is None:
            raise RuntimeError('Falha ao chamar o serviço /get_map')
        return res

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

    def run(
        self,
        algorithm: str = 'astar',
        delay: float = 0.03,
        show_ascii: bool = True,
    ) -> bool:
        """Fetch the map, solve the optimal route, and execute all steps."""
        self.wait_for_services()
        self.get_logger().info('=== PARTE 1: Navegação com Mapa ===')
        map_res = self.fetch_map()
        grid, start, target = parse_flattened_grid(
            map_res.occupancy_grid_flattened,
            map_res.occupancy_grid_shape,
        )

        rows, cols = len(grid), len(grid[0])
        self.get_logger().info(
            f'Mapa recebido ({rows}x{cols}) | Início (r): {start} | '
            f'Alvo (t): {target}'
        )

        t0 = time.perf_counter()
        result = solve_grid(grid, start, target, algorithm=algorithm)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.get_logger().info(
            f'Algoritmo: {result.algorithm.upper()} | '
            f'Rota ótima: {len(result.directions)} passos | '
            f'Nós expandidos: {result.nodes_expanded} | '
            f'Tempo de busca: {elapsed_ms:.2f} ms'
        )

        if show_ascii:
            ascii_map = format_grid_ascii(grid, result.path)
            self.get_logger().info('Mapa com rota ótima (*):\n' + ascii_map)

        last_res: Optional[MoveCmd.Response] = None
        for step_idx, direction in enumerate(result.directions, start=1):
            last_res = self.send_move(direction)
            if not last_res.success:
                pos = [int(v) for v in last_res.robot_pos]
                self.get_logger().error(
                    f'Movimento falhou no passo '
                    f'{step_idx}/{len(result.directions)} ({direction}). '
                    f'Posição atual: {pos}'
                )
                return False
            if delay > 0.0:
                time.sleep(delay)

        if start == target:
            reached = True
            final_pos = list(start)
        else:
            reached = (
                last_res is not None
                and tuple(last_res.robot_pos) == tuple(last_res.target_pos)
            )
            final_pos = (
                [int(v) for v in last_res.robot_pos]
                if last_res is not None
                else list(start)
            )

        if reached:
            self.get_logger().info(
                f'Alvo alcançado com sucesso em {len(result.directions)} '
                f'passos! Posição final: {final_pos}'
            )
            return True

        self.get_logger().error('Rota concluída sem atingir o alvo!')
        return False


def main(args=None) -> None:
    """CLI entrypoint for the Part 1 autonomous navigator node."""
    parser = argparse.ArgumentParser(
        description='Parte 1 — Navegação autônoma com acesso ao mapa '
        '(/get_map).'
    )
    parser.add_argument(
        '--algorithm',
        choices=['astar', 'bfs'],
        default='astar',
        help='Algoritmo de busca em grafo (padrão: astar).',
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.03,
        help='Atraso em segundos entre passos para visualização '
        '(padrão: 0.03).',
    )
    parser.add_argument(
        '--no-ascii',
        action='store_true',
        help='Não imprime o mapa ASCII no terminal.',
    )
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = Part1Navigator()
    success = False
    try:
        success = node.run(
            algorithm=parsed.algorithm,
            delay=parsed.delay,
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
