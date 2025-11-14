import rclpy
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

class InteractorWrapper(LifecycleNode):
    """
    ROS2 Lifecycle Node Wrapper for Interactors
    Wraps and manages any type of interactor within a ROS2 lifecycle node.
    Publication is based on returns of interactor methods and set up and 
    destroyed according to the active behavior passed from the Behavior Manager.
    Service calls allow single-step execution of interactor logic to refresh the state.
    
    """
    def __init__(self):
        super().__init__('interactor_wrapper')
        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )