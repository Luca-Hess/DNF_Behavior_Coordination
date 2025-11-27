import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.animation import FuncAnimation


class RobotSimulationVisualizer:
    """Visualize the robot, gripper, object and node activity in 3D."""
    def __init__(self, floor_size=25, behavior_chain=None):
        # Set up the figure and 3D axes
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.floor_size = floor_size

        self.target_tube = None
        self.target_floor_marker = None
        self.drop_off_tube = None
        self.drop_off_floor_marker = None

        # Better camera position for initial view
        self.ax.view_init(elev=30, azim=30)  # Set camera angle

        # Setup plot limits and labels with consistent spacing
        self.ax.set_xlim(-floor_size / 2, floor_size / 2)  # Center at origin
        self.ax.set_ylim(-floor_size / 2, floor_size / 2)  # Center at origin
        self.ax.set_zlim(0, floor_size / 2)  # Only show positive z
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Robot Behavior Simulation')

        # Create the floor (a simple plane) - centered at origin
        floor_x = np.array([-floor_size / 2, floor_size / 2, floor_size / 2, -floor_size / 2])
        floor_y = np.array([-floor_size / 2, -floor_size / 2, floor_size / 2, floor_size / 2])
        floor_z = np.zeros(4)
        verts = [list(zip(floor_x, floor_y, floor_z))]
        self.floor = Poly3DCollection(verts, alpha=0.3, color='gray')
        self.ax.add_collection3d(self.floor)

        # Initial objects (will be updated later)
        self.robot = None
        self.gripper = None
        self.target_obj = None
        self.drop_off = None


    def update(self, state, interactors=None):
        """Update the visualization with the current state."""
        # Extract positions from state
        robot_position = state.get('robot_pos', {})
        gripper_position = state.get('gripper_pos', {})

        # Get target position if available
        target_position = state.get('target_position', {})

        drop_off_position = None
        # Get drop-off position if available
        if interactors is not None and "transport_target" in interactors.perception.objects:
            drop_off_position = interactors.perception.objects["transport_target"]['location'].tolist()

            if self.drop_off is None:
                self.drop_off = self.ax.plot([drop_off_position[0]],
                                             [drop_off_position[1]],
                                             [drop_off_position[2]],
                                             'o', markersize=8, color='green')[0]
                self._create_drop_off_tube(drop_off_position)
            else:
                self.drop_off.set_data([drop_off_position[0]], [drop_off_position[1]])
                self.drop_off.set_3d_properties([drop_off_position[2]])
                self._update_drop_off_tube(drop_off_position)

        # Create or update robot visualization
        if self.robot is None:
            self.robot = self.ax.plot([robot_position[0]], [robot_position[1]], [robot_position[2]],
                                      'o', markersize=10, color='blue')[0]
        else:
            self.robot.set_data([robot_position[0]], [robot_position[1]])
            self.robot.set_3d_properties([robot_position[2]])

        # Create or update gripper visualization
        if self.gripper is None:
            self.gripper = self.ax.plot([gripper_position[0]], [gripper_position[1]], [gripper_position[2]],
                                        'o', markersize=6, color='cyan')[0]
        else:
            self.gripper.set_data([gripper_position[0]], [gripper_position[1]])
            self.gripper.set_3d_properties([gripper_position[2]])

        # Create or update target object and its tube
        if target_position is not None:
            if self.target_obj is None:
                self.target_obj = self.ax.plot([target_position[0]], [target_position[1]], [target_position[2]],
                                               'o', markersize=8, color='red')[0]

            else:
                # Update target position
                self.target_obj.set_data([target_position[0]], [target_position[1]])
                self.target_obj.set_3d_properties([target_position[2]])


        # Create or update drop-off object and its tube
        if drop_off_position is not None:
            if self.drop_off is None:
                self.drop_off = self.ax.plot([drop_off_position[0]],
                                             [drop_off_position[1]],
                                             [drop_off_position[2]],
                                             'o', markersize=8, color='green')[0]

                # Create the tube for drop-off
                self._create_drop_off_tube(drop_off_position)
            else:
                # Update drop-off position (FIXED)
                self.drop_off.set_data([drop_off_position[0]], [drop_off_position[1]])
                self.drop_off.set_3d_properties([drop_off_position[2]])

                # Update tube position
                self._update_drop_off_tube(drop_off_position)


        # Return artists for animation
        artists = [self.floor]
        if self.robot is not None:
            artists.append(self.robot)
        if self.gripper is not None:
            artists.append(self.gripper)
        if self.target_obj is not None:
            artists.append(self.target_obj)
        if hasattr(self, 'tube') and self.tube is not None:
            artists.append(self.tube)
        if self.drop_off is not None:
            artists.append(self.drop_off)
        if hasattr(self, 'drop_off_tube') and self.drop_off_tube is not None:
            artists.append(self.drop_off_tube)
        if hasattr(self, 'drop_off_floor_marker') and self.drop_off_floor_marker is not None:
            artists.append(self.drop_off_floor_marker)

        return artists

    def _create_tube(self, position):
        """Create a transparent tube from target to floor."""
        x, y, z = position
        # Draw a line from object to floor
        self.tube = self.ax.plot([x, x], [y, y], [z, 0],
                                 linestyle='--', linewidth=2,
                                 color='red', alpha=0.4)[0]

        # Add a small circle on the floor to mark position
        radius = 0.3
        theta = np.linspace(0, 2 * np.pi, 32)
        circle_x = x + radius * np.cos(theta)
        circle_y = y + radius * np.sin(theta)
        circle_z = np.zeros_like(theta)
        self.floor_marker = self.ax.plot(circle_x, circle_y, circle_z,
                                         color='red', alpha=0.5)[0]

    def _update_tube(self, position):
        """Update the tube position."""
        x, y, z = position
        self.tube.set_data([x, x], [y, y])
        self.tube.set_3d_properties([z, 0])

        # Update floor marker
        radius = 0.3
        theta = np.linspace(0, 2 * np.pi, 32)
        circle_x = x + radius * np.cos(theta)
        circle_y = y + radius * np.sin(theta)
        circle_z = np.zeros_like(theta)
        self.floor_marker.set_data(circle_x, circle_y)
        self.floor_marker.set_3d_properties(circle_z)

    def _update_robot(self, position):
        """Update the robot disk position."""
        x, y, z = position
        if self.robot is not None:
            self.robot.remove()

        # Create robot as disk
        radius = 1.0
        u = np.linspace(0, 2 * np.pi, 32)
        v = np.linspace(0, np.pi, 16)
        x_points = radius * np.outer(np.cos(u), np.sin(v)) + x
        y_points = radius * np.outer(np.sin(u), np.sin(v)) + y
        z_points = 0.2 * np.outer(np.ones(np.size(u)), np.cos(v)) + 0.2

        self.robot = self.ax.plot_surface(x_points, y_points, z_points, color='blue', alpha=0.7)

    def _update_gripper(self, robot_pos, gripper_pos):
        """Update the gripper tube."""
        if self.gripper is not None:
            self.gripper.remove()

        # Create tube from robot to gripper
        x_rob, y_rob, _ = robot_pos
        x_grip, y_grip, z_grip = gripper_pos

        # Cylinder parameters
        radius = 0.2
        height = z_grip

        # Create points for cylinder
        u = np.linspace(0, 2 * np.pi, 16)
        h = np.linspace(0, height, 10)
        X = radius * np.outer(np.cos(u), np.ones_like(h)) + x_rob
        Y = radius * np.outer(np.sin(u), np.ones_like(h)) + y_rob
        Z = np.outer(np.ones_like(u), h) + 0.4  # Start slightly above the robot

        self.gripper = self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.7)

        # Add a sphere at the gripper end
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 16)
        x_points = 0.4 * np.outer(np.cos(u), np.sin(v)) + x_grip
        y_points = 0.4 * np.outer(np.sin(u), np.sin(v)) + y_grip
        z_points = 0.4 * np.outer(np.ones(np.size(u)), np.cos(v)) + z_grip

        self.ax.plot_surface(x_points, y_points, z_points, color='red', alpha=0.7)


    def _update_target_object(self, position):
        """Update the target object (cube)."""
        if self.target_obj is not None:
            self.target_obj.remove()

        x, y, z = position
        size = 0.8

        # Create a cube at the target position
        vertices = [
            [x - size / 2, y - size / 2, z - size / 2],
            [x + size / 2, y - size / 2, z - size / 2],
            [x + size / 2, y + size / 2, z - size / 2],
            [x - size / 2, y + size / 2, z - size / 2],
            [x - size / 2, y - size / 2, z + size / 2],
            [x + size / 2, y - size / 2, z + size / 2],
            [x + size / 2, y + size / 2, z + size / 2],
            [x - size / 2, y + size / 2, z + size / 2]
        ]

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]

        self.target_obj = Poly3DCollection(faces, alpha=0.5, linewidth=1, edgecolor='k')
        self.target_obj.set_facecolor('orange')
        self.ax.add_collection3d(self.target_obj)

    def _create_drop_off_tube(self, position):
        """Create a transparent tube from drop-off point to floor."""
        x, y, z = position
        # Draw a line from object to floor
        self.drop_off_tube = self.ax.plot([x, x], [y, y], [z, 0],
                                          linestyle='--', linewidth=2,
                                          color='green', alpha=0.4)[0]

        # Add a small circle on the floor to mark position
        radius = 0.3
        theta = np.linspace(0, 2 * np.pi, 32)
        circle_x = x + radius * np.cos(theta)
        circle_y = y + radius * np.sin(theta)
        circle_z = np.zeros_like(theta)
        self.drop_off_floor_marker = self.ax.plot(circle_x, circle_y, circle_z,
                                                  color='green', alpha=0.5)[0]

    def _update_drop_off_tube(self, position):
        """Update the drop-off tube position."""
        x, y, z = position
        self.drop_off_tube.set_data([x, x], [y, y])
        self.drop_off_tube.set_3d_properties([z, 0])

        # Update floor marker
        radius = 0.3
        theta = np.linspace(0, 2 * np.pi, 32)
        circle_x = x + radius * np.cos(theta)
        circle_y = y + radius * np.sin(theta)
        circle_z = np.zeros_like(theta)
        self.drop_off_floor_marker.set_data(circle_x, circle_y)
        self.drop_off_floor_marker.set_3d_properties(circle_z)
