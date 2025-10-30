import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Pose, PoseArray, Twist, Point
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
import threading
import time
from queue import Queue, Empty

class ROSInteractorNode(Node):
    """ROS node that manages dynamic subscriptions/publications for DNF behaviors"""
    
    def __init__(self):
        super().__init__('dnf_interactor_node')
        
        # Data storage for interactors
        self.shared_data = {
            'objects': {},
            'robot_pose': None,
            'commands': Queue(),
            'search_requests': Queue(),
            'active_searches': set(),  # added: used by start/stop_object_search

        }
        
        # Dynamic subscription management
        self.active_subscriptions = {}
        self._pubs = []
        
        # Create core publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Core robot pose subscription (always active)
        self.robot_sub = self.create_subscription(
            Pose, '/robot/pose', self._on_robot_pose, 10)
        
        # Sensor subscriptions for object detection (always active)
        self.rgb_sub = self.create_subscription(
            Image, '/camera/image', self._on_rgb_image, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth_image', self._on_depth_image, 10)
        

        # Subscribe to ground truth object poses from Gazebo (for demo)
        self.gazebo_poses = {}
        self.pose_sub = self.create_subscription(
            PoseArray, '/model/dynamic_pose', self._on_model_poses, 10)

        # Command processing timer
        self.timer = self.create_timer(0.01, self._process_commands)

        # Object detection timer (20Hz)
        self.detection_timer = self.create_timer(0.05, self._run_object_detection)
        
        
        self.get_logger().info("ROS Interactor Node started")
    

    def _on_model_poses(self, msg):
        """Receive ground truth pose from Gazebo"""
        if msg.poses:
        # For demo: assume pose[0] = cricket_ball, pose[1] = other_object, etc.
            self.gazebo_poses['cricket_ball'] = {
                'position': [msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z],
                'orientation': [msg.poses[0].orientation.x, msg.poses[0].orientation.y, 
                            msg.poses[0].orientation.z, msg.poses[0].orientation.w],
                'timestamp': time.time()
            }

    def _on_rgb_image(self, msg):
        """Store RGB image for object detection"""
        # For now, just store the message
        # In real implementation, would run vision processing here
        pass
    
    def _on_depth_image(self, msg):
        """Store depth image for object detection"""
        # For now, just store the message
        # In real implementation, would run vision processing here
        pass


    def start_object_search(self, object_name):
        """Start actively searching for specific object"""
        if 'active_searches' not in self.shared_data:
            self.shared_data['active_searches'] = set()
        if object_name not in self.shared_data['active_searches']:
            self.get_logger().info(f"Starting active search for: {object_name}")
            self.shared_data['active_searches'].add(object_name)
    
    def stop_object_search(self, object_name):
        """Stop searching for specific object"""
        if object_name in self.shared_data['active_searches']:
            self.get_logger().info(f"Stopping search for: {object_name}")
            self.shared_data['active_searches'].remove(object_name)
            
            # Clean up detected object data
            if object_name in self.shared_data['objects']:
                del self.shared_data['objects'][object_name]
    
    def _run_object_detection(self):
        """Run object detection for actively searched objects"""
        for object_name in self.shared_data['active_searches'].copy():
            # For demo: use ground truth data from Gazebo
            if object_name in self.gazebo_poses:
                gazebo_data = self.gazebo_poses[object_name]
                
                # Only use recent data (within 1 second)
                if time.time() - gazebo_data['timestamp'] < 1.0:
                    self.shared_data['objects'][object_name] = gazebo_data
                    self.get_logger().debug(f"Detected {object_name} at {gazebo_data['position']}")
            
            # In real implementation, this would run:
            # - YOLO/other object detection on RGB image
            # - Depth lookup for 3D position
            # - Pose estimation
            # - Update self.shared_data['objects'][object_name]
    

    def _on_robot_pose(self, msg):
        """Update robot pose"""
        self.shared_data['robot_pose'] = {
            'position': [msg.position.x, msg.position.y, msg.position.z],
            'orientation': [msg.orientation.x, msg.orientation.y, 
                           msg.orientation.z, msg.orientation.w],
            'timestamp': time.time()
        }
    
    def _process_commands(self):
        """Process commands from DNF behaviors"""
        try:
            while not self.shared_data['commands'].empty():
                cmd = self.shared_data['commands'].get_nowait()
                
                if cmd['type'] == 'move_to':
                    self._execute_move_command(cmd['data'])
                elif cmd['type'] == 'search_object':
                    self.start_object_search(cmd['data'])
                elif cmd['type'] == 'stop_search':
                    self.stop_object_search(cmd['data'])
                    
        except Empty:
            pass
    
    
    def _execute_move_command(self, target_position):
        """Convert target position to velocity command"""
        if self.shared_data['robot_pose'] is None:
            return
            
        robot_pos = self.shared_data['robot_pose']['position']
        
        # Simple proportional control
        dx = target_position[0] - robot_pos[0]
        dy = target_position[1] - robot_pos[1]
        
        # Limit velocity
        max_vel = 0.5
        vel_x = max(-max_vel, min(max_vel, dx * 0.5))
        vel_y = max(-max_vel, min(max_vel, dy * 0.5))
        
        twist = Twist()
        twist.linear.x = vel_x
        twist.linear.y = vel_y
        self.cmd_vel_pub.publish(twist)

class PerceptionInteractor:
    """Perception interface for DNF behaviors"""
    
    def __init__(self, ros_node):
        self.ros_node = ros_node
        self.active_objects = set()
    
    def find_object(self, target_name):
        """Find object and return (found, location)"""
        # Dynamically subscribe if not already searching
        if target_name not in self.active_objects:
            self.ros_node.shared_data['commands'].put({
                'type': 'search_object', 
                'data': target_name
            })
            self.active_objects.add(target_name)
        
        # Check if object has been detected
        objects = self.ros_node.shared_data['objects']
        if target_name in objects:
            obj_data = objects[target_name]
            # Check if data is recent (within 1 second)
            if time.time() - obj_data['timestamp'] < 1.0:
                return True, obj_data['position']
        
        return False, None
    
    def stop_tracking(self, target_name):
        """Stop tracking specific object"""
        if target_name in self.active_objects:
            self.ros_node.shared_data['commands'].put({
                'type': 'stop_search',
                'data': target_name
            })
            self.active_objects.remove(target_name)

class MovementInteractor:
    """Movement interface for DNF behaviors"""
    
    def __init__(self, ros_node):
        self.ros_node = ros_node
    
    def get_position(self):
        """Get current robot position"""
        
        if self.ros_node.shared_data['robot_pose'] is None:
            return (0.0, 0.0, 0.0)
        
        pos = self.ros_node.shared_data['robot_pose']['position']
        return pos
    
    
    def move_to(self, target_location):
        """Move robot to target location"""
        if hasattr(target_location, 'tolist'):
            target_pos = target_location.tolist()
        else:
            target_pos = list(target_location)
        
        self.ros_node.shared_data['commands'].put({
            'type': 'move_to',
            'data': target_pos
        })

class RobotInteractors:
    """Main interactors class compatible with DNF behaviors"""
    
    def __init__(self):
        # Initialize ROS if not already done
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS node
        self.ros_node = ROSInteractorNode()
        
        # Create interactor interfaces
        self.perception = PerceptionInteractor(self.ros_node)
        self.movement = MovementInteractor(self.ros_node)
        
        # Start ROS spinning in separate thread
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)
        self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.ros_thread.start()
        
        print("ROS-based RobotInteractors initialized")
    
    def shutdown(self):
        """Clean shutdown"""
        self.executor.shutdown()
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()