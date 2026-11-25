import os
import random
import threading

from ament_index_python.packages import get_package_share_directory
from cg_interfaces.msg import RobotSensors
from cg_interfaces.srv import GetMap, MoveCmd, Reset
import pygame
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .Maze import Maze
from .Robot import Robot
from .Utils.Csv import load_from_csv
from .Utils.Finders import find
from .Utils.Grid import flatten


class Game(Node):
    def __init__(self, map_path, resolution=720):
        super().__init__('Culling_Games')
        self.map_path = map_path
        self.maps_dir = os.path.join(get_package_share_directory('cg'), 'maps')
        self.resolution = resolution

        self.move_cmd_service = self.create_service(
            MoveCmd, 'move_command', self.handle_move_cmd
        )
        self.get_map_service = self.create_service(
            GetMap, 'get_map', self.handle_map_request
        )
        self.reset_service = self.create_service(
            Reset, 'reset', self.handle_reset_request
        )

        self.publisher_ = self.create_publisher(
            RobotSensors, '/culling_games/robot_sensors', qos_profile_sensor_data
        )
        self.timer = self.create_timer(0.01, self.publish_sensor_data)

        initial_maze_config = load_from_csv(self.map_path)
        self.maze = Maze(initial_maze_config, self.resolution)
        self.robot = Robot(self.maze)

        self.running = False
        self.win = False
        self.win_handled = False

        self.reset_requested = False
        self.reset_done_event = threading.Event()
        self.state_lock = threading.Lock()

    def handle_reset_request(self, request, response):
        with self.state_lock:
            self.get_logger().info(
                f'Reset request received: is_random={request.is_random}, '
                f'map_name={request.map_name}'
            )
            if request.is_random:
                available_maps = [
                    f for f in os.listdir(self.maps_dir) if f.endswith('.csv')
                ]
                if available_maps:
                    new_map_name = random.choice(available_maps)
                    self.map_path = os.path.join(self.maps_dir, new_map_name)
                    self.get_logger().info(
                        f'Randomly selected new map: {new_map_name}'
                    )
            elif request.map_name:
                new_map_path = os.path.join(self.maps_dir, request.map_name)
                if os.path.exists(new_map_path):
                    self.map_path = new_map_path
                    self.get_logger().info(f'Switching to map: {request.map_name}')
                else:
                    self.get_logger().error(
                        f'Requested map not found: {request.map_name}'
                    )
                    response.success = False
                    response.loaded_map_name = os.path.basename(self.map_path)
                    return response

            self.reset_done_event.clear()
            self.reset_requested = True
            is_running = self.running
            response.loaded_map_name = os.path.basename(self.map_path)

        if is_running:
            completed = self.reset_done_event.wait(timeout=5.0)
            response.success = completed
        else:
            response.success = True
        return response

    def publish_sensor_data(self):
        msg = RobotSensors()
        with self.state_lock:
            surroundings = self.robot.check_surroundings()
        msg.up = surroundings['up']
        msg.down = surroundings['down']
        msg.left = surroundings['left']
        msg.right = surroundings['right']
        msg.up_left = surroundings['up_left']
        msg.up_right = surroundings['up_right']
        msg.down_left = surroundings['down_left']
        msg.down_right = surroundings['down_right']
        self.publisher_.publish(msg)

    def update(self):
        with self.state_lock:
            if self.reset_requested:
                self.get_logger().info(
                    f'Resetting game with map: {os.path.basename(self.map_path)}'
                )
                new_maze_config = load_from_csv(self.map_path)
                self.maze = Maze(new_maze_config, self.resolution)
                self.robot = Robot(self.maze)
                self.win = False
                self.win_handled = False
                self.reset_requested = False
                self.reset_done_event.set()

        self.handle_input()
        maze_ref = None
        grid_snapshot = None
        with self.state_lock:
            if not self.win:
                maze_ref = self.maze
                grid_snapshot = [row[:] for row in self.maze.occupancy_grid]
        if maze_ref is not None:
            maze_ref.draw(grid_snapshot)

    def handle_map_request(self, request, response):
        with self.state_lock:
            occ_grid = self.maze.get_occupancy_grid()
        response.occupancy_grid_flattened = flatten(occ_grid)
        response.occupancy_grid_shape = len(occ_grid), len(occ_grid[0])
        return response

    def handle_move_cmd(self, request, response):
        with self.state_lock:
            if self.win:
                self.get_logger().warn(
                    'Game has already been won, ignoring move command.'
                )
                response.success = False
                response.robot_pos = self.robot.pos
                # Target no longer exists, so use robot's final position.
                response.target_pos = self.robot.pos
                return response

            direction = request.direction.lower()
            target_pos = find(self.maze.get_occupancy_grid(), 't')
            response.success = self.robot.move(direction)
            if target_pos == self.robot.pos:
                self.win = True
            response.robot_pos = self.robot.pos
            response.target_pos = target_pos
        return response

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                with self.state_lock:
                    self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    direction = 'left'
                elif event.key == pygame.K_j:
                    direction = 'down'
                elif event.key == pygame.K_k:
                    direction = 'up'
                elif event.key == pygame.K_l:
                    direction = 'right'
                else:
                    continue
                with self.state_lock:
                    if self.win:
                        continue
                    target_pos = find(self.maze.get_occupancy_grid(), 't')
                    self.robot.move(direction)
                    if target_pos == self.robot.pos:
                        self.win = True

    def run(self):
        pygame.init()
        clock = pygame.time.Clock()
        with self.state_lock:
            self.running = True
        try:
            while rclpy.ok():
                with self.state_lock:
                    if not self.running:
                        break
                self.update()
                pygame.display.flip()

                win_maze = None
                with self.state_lock:
                    if self.win and not self.win_handled:
                        self.win_handled = True
                        win_maze = self.maze
                if win_maze is not None:
                    win_maze.win()
                clock.tick(60)
        finally:
            with self.state_lock:
                self.running = False
            self.reset_done_event.set()
            pygame.quit()
