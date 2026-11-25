"""Keyboard teleoperation node for the Culling Games maze simulator."""

import os
import select
import sys
import termios
import tty

from cg_interfaces.srv import MoveCmd, Reset

import rclpy
from rclpy.node import Node

msg = """
Reading from the keyboard and calling /move_command and /reset!
---------------------------
Moving around:
   w / k / Up arrow    : up
   s / j / Down arrow  : down
   a / h / Left arrow  : left
   d / l / Right arrow : right

r : reset the current board
n : start a new random board

CTRL-C to quit
"""


class TeleopKeyboard(Node):
    """ROS 2 node that sends keyboard movement and reset service requests."""

    def __init__(self):
        """Initialize service clients and wait for simulator services."""
        super().__init__('teleop_keyboard')
        self.move_cli = self.create_client(MoveCmd, '/move_command')
        self.reset_cli = self.create_client(Reset, '/reset')

        while rclpy.ok() and not self.move_cli.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Move service not available, waiting again...'
            )
        while rclpy.ok() and not self.reset_cli.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Reset service not available, waiting again...'
            )

        self.move_req = MoveCmd.Request()

    def send_move_request(self, direction):
        """Send a move request for the given direction and wait up to 5s."""
        self.move_req.direction = direction
        self.future = self.move_cli.call_async(self.move_req)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=5.0)
        if not self.future.done():
            self.get_logger().warning('Move service request timed out')
            return None
        return self.future.result()

    def send_reset_request(self, is_random=False):
        """Send a board reset request and wait up to 5s."""
        reset_req = Reset.Request()
        reset_req.is_random = is_random
        self.future = self.reset_cli.call_async(reset_req)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=5.0)
        if not self.future.done():
            self.get_logger().warning('Reset service request timed out')
            return None
        return self.future.result()


def get_key(settings):
    """Read a single keypress or 3-byte arrow escape sequence."""
    fd = sys.stdin.fileno()
    tty.setraw(fd)
    try:
        key = os.read(fd, 1).decode('utf-8', errors='ignore')
        if key == '\x1b':
            for _ in range(2):
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    break
                key += os.read(fd, 1).decode('utf-8', errors='ignore')
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def _log_reset_result(node, response, is_random):
    """Log the outcome of a reset service call."""
    if response is not None and response.success:
        action = 'loaded new random map' if is_random else 'reset board to map'
        node.get_logger().info(
            f'Successfully {action}: {response.loaded_map_name}'
        )
    elif response is None:
        node.get_logger().error(
            'Reset request timed out or failed without a response'
        )
    else:
        detail = 'load new random map' if is_random else 'reset the board'
        node.get_logger().info(f'Failed to {detail}')


def main(args=None):
    """Run the keyboard teleoperation loop."""
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init(args=args)
    teleop_node = None
    try:
        teleop_node = TeleopKeyboard()
        print(msg)
        while rclpy.ok():
            key = get_key(settings)
            direction = None
            if key in ['w', 'k', '\x1b[A']:
                direction = 'up'
            elif key in ['s', 'j', '\x1b[B']:
                direction = 'down'
            elif key in ['a', 'h', '\x1b[D']:
                direction = 'left'
            elif key in ['d', 'l', '\x1b[C']:
                direction = 'right'
            elif key == 'r':
                response = teleop_node.send_reset_request(is_random=False)
                _log_reset_result(teleop_node, response, is_random=False)
            elif key == 'n':
                response = teleop_node.send_reset_request(is_random=True)
                _log_reset_result(teleop_node, response, is_random=True)
            elif key == '\x03':
                break

            if direction:
                response = teleop_node.send_move_request(direction)
                if response is not None and response.success:
                    teleop_node.get_logger().info(
                        f'Successfully moved {direction}'
                    )
                elif response is None:
                    teleop_node.get_logger().error(
                        f'Move {direction} timed out or failed'
                    )
                else:
                    teleop_node.get_logger().info(
                        f'Failed to move {direction}'
                    )
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        if teleop_node is not None:
            teleop_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
