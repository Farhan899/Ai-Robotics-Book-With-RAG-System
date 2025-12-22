# Hands-On: NVIDIA Isaac AI Perception and Training

## Exercise 1: Setting Up Isaac Sim Environment

In this exercise, you'll set up the Isaac Sim environment and launch a basic simulation.

### Prerequisites
- NVIDIA GPU with CUDA support
- Isaac Sim installed (Omniverse Isaac Sim app)
- Isaac ROS packages installed

### Steps

1. Verify Isaac Sim installation:
   ```bash
   # Check if Omniverse launcher is available
   # Launch Isaac Sim from Omniverse Launcher
   ```

2. Verify Isaac ROS installation:
   ```bash
   # Check available Isaac ROS packages
   ros2 pkg list | grep isaac_ros
   ```

3. Launch Isaac Sim with a basic scene:
   - Open Isaac Sim from Omniverse Launcher
   - Select "Isaac Sim" app
   - Load a basic scene (e.g., "Simple Room")

## Exercise 2: Creating a Robot in Isaac Sim

Now you'll add a robot to the Isaac Sim environment.

### Steps

1. Create a new USD stage in Isaac Sim:
   - File → New Stage
   - Save the stage to your project directory

2. Import a robot model:
   - Use the "Import Robot" extension
   - Or manually create robot links and joints using USD primitives

3. Example robot creation using Python API:
   ```python
   import omni
   from pxr import UsdGeom, Usd, Sdf
   import carb

   # Get the current stage
   stage = omni.usd.get_context().get_stage()

   # Create a robot prim
   robot_prim = stage.DefinePrim("/World/MyRobot", "Xform")
   robot_prim.GetAttribute("xformOp:translate").Set((0.0, 0.0, 1.0))

   # Create a base link
   base_link = stage.DefinePrim("/World/MyRobot/BaseLink", "Xform")

   # Add visual and collision geometries
   visual_geom = stage.DefinePrim("/World/MyRobot/BaseLink/Visual", "Cube")
   collision_geom = stage.DefinePrim("/World/MyRobot/BaseLink/Collision", "Cube")
   ```

4. Add sensors to the robot:
   - Camera sensor for perception
   - IMU for orientation
   - LIDAR for navigation

## Exercise 3: Synthetic Data Generation

Now you'll set up a synthetic data generation pipeline.

### Steps

1. Configure domain randomization:
   - Open the "Synthetic Data" extension
   - Add randomization nodes for:
     - Lighting conditions
     - Object textures
     - Camera parameters
     - Physical properties

2. Example domain randomization setup:
   ```python
   # Randomize material properties
   from omni.isaac.synthetic_utils import SyntheticDataHelper

   # Create material randomizer
   material_randomizer = RandomRGLightAPI.get_randomizer()
   material_randomizer.randomize_materials = True
   material_randomizer.material_variations = {
       "roughness": (0.1, 0.9),
       "metallic": (0.0, 0.2),
       "specular": (0.5, 1.0)
   }
   ```

3. Set up data capture:
   - Configure RGB image capture
   - Set up segmentation masks
   - Capture depth information
   - Generate bounding boxes for objects

4. Run the synthetic data generation:
   - Execute the randomization script
   - Capture data from multiple viewpoints
   - Generate annotations automatically

## Exercise 4: Isaac ROS VSLAM Implementation

Now you'll implement Visual SLAM using Isaac ROS packages.

### Steps

1. Create a ROS 2 launch file for VSLAM:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import ComposableNodeContainer
   from launch_ros.descriptions import ComposableNode


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time')
       declare_use_sim_time_cmd = DeclareLaunchArgument(
           'use_sim_time',
           default_value='True',
           description='Use simulation (Gazebo) clock if true')

       # Create a container for VSLAM nodes
       vslam_container = ComposableNodeContainer(
           name='vslam_container',
           namespace='',
           package='rclcpp_components',
           executable='component_container_mt',
           composable_node_descriptions=[
               ComposableNode(
                   package='isaac_ros_visual_slam',
                   plugin='isaac_ros::visual_slam::VisualSlamNode',
                   name='visual_slam',
                   parameters=[{
                       'use_sim_time': use_sim_time,
                       'enable_rectified_pose': True,
                       'map_frame': 'map',
                       'odom_frame': 'odom',
                       'base_frame': 'base_link',
                       'publish_odom_tf': True
                   }],
                   remappings=[
                       ('/visual_slam/tracking/feature0/image', '/camera/image_raw'),
                       ('/visual_slam/tracking/feature0/camera_info', '/camera/camera_info'),
                       ('/visual_slam/tracking/imu', '/imu/data')
                   ]
               )
           ],
           output='screen'
       )

       return LaunchDescription([
           declare_use_sim_time_cmd,
           vslam_container
       ])
   ```

2. Launch the VSLAM pipeline:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 launch vslam_launch.py
   ```

3. Visualize the results in RViz:
   - Add a RobotModel display
   - Add a Map display for the generated map
   - Add TF display to visualize the robot pose

## Exercise 5: Nav2 Configuration for Bipedal Navigation

Now you'll configure Nav2 for humanoid navigation in simulation.

### Steps

1. Create a Nav2 configuration file `humanoid_nav2_params.yaml`:
   ```yaml
   amcl:
     ros__parameters:
       use_sim_time: True
       alpha1: 0.2
       alpha2: 0.2
       alpha3: 0.2
       alpha4: 0.2
       alpha5: 0.2
       base_frame_id: "base_link"
       beam_skip_distance: 0.5
       beam_skip_error_threshold: 0.9
       beam_skip_threshold: 0.3
       do_beamskip: false
       global_frame_id: "map"
       lambda_short: 0.1
       laser_likelihood_max_dist: 2.0
       laser_max_range: 100.0
       laser_min_range: -1.0
       laser_model_type: "likelihood_field"
       max_beams: 60
       max_particles: 2000
       min_particles: 500
       odom_frame_id: "odom"
       pf_err: 0.05
       pf_z: 0.99
       recovery_alpha_fast: 0.0
       recovery_alpha_slow: 0.0
       resample_interval: 1
       robot_model_type: "nav2_amcl::DifferentialMotionModel"
       save_pose_rate: 0.5
       sigma_hit: 0.2
       tf_broadcast: true
       transform_tolerance: 1.0
       update_min_a: 0.2
       update_min_d: 0.25
       z_hit: 0.5
       z_max: 0.05
       z_rand: 0.5
       z_short: 0.05

   bt_navigator:
     ros__parameters:
       use_sim_time: True
       global_frame: map
       robot_base_frame: base_link
       odom_topic: /odom
       bt_loop_duration: 10
       default_server_timeout: 20
       enable_groot_monitoring: True
       groot_zmq_publisher_port: 1666
       groot_zmq_server_port: 1667
       default_nav_through_poses_bt_xml: $(find package_name)/behavior_trees/navigate_w_replanning_and_recovery.xml
       default_nav_to_pose_bt_xml: $(find package_name)/behavior_trees/navigate_w_replanning_and_recovery.xml
       plugin_lib_names:
       - nav2_compute_path_to_pose_action_bt_node
       - nav2_compute_path_through_poses_action_bt_node
       - nav2_follow_path_action_bt_node
       - nav2_back_up_action_bt_node
       - nav2_spin_action_bt_node
       - nav2_wait_action_bt_node
       - nav2_assisted_teleop_action_bt_node
       - nav2_clear_costmap_service_bt_node
       - nav2_is_stuck_condition_bt_node
       - nav2_goal_reached_condition_bt_node
       - nav2_goal_updated_condition_bt_node
       - nav2_initial_pose_received_condition_bt_node
       - nav2_reinitialize_global_localization_service_bt_node
       - nav2_rate_controller_bt_node
       - nav2_distance_controller_bt_node
       - nav2_speed_controller_bt_node
       - nav2_truncate_path_action_bt_node
       - nav2_truncate_path_local_action_bt_node
       - nav2_goal_updater_node_bt_node
       - nav2_recovery_node_bt_node
       - nav2_pipeline_sequence_bt_node
       - nav2_round_robin_node_bt_node
       - nav2_transform_available_condition_bt_node
       - nav2_time_expired_condition_bt_node
       - nav2_path_expiring_timer_condition
       - nav2_distance_traveled_condition_bt_node
       - nav2_single_trigger_bt_node
       - nav2_is_battery_low_condition_bt_node
       - nav2_navigate_through_poses_action_bt_node
       - nav2_navigate_to_pose_action_bt_node
       - nav2_remove_passed_goals_action_bt_node
       - nav2_planner_selector_bt_node
       - nav2_controller_selector_bt_node
       - nav2_goal_checker_selector_bt_node

   controller_server:
     ros__parameters:
       use_sim_time: True
       controller_frequency: 20.0
       min_x_velocity_threshold: 0.001
       min_y_velocity_threshold: 0.5
       min_theta_velocity_threshold: 0.001
       progress_checker_plugin: "progress_checker"
       goal_checker_plugin: "goal_checker"
       controller_plugins: ["FollowPath"]

       # Humanoid-specific controller
       FollowPath:
         plugin: "nav2_mppi_controller::MppiController"
         time_steps: 24
         control_freq: 20.0
         horizon: 1.2
         Q: [2.0, 2.0, 0.5, 0.0, 0.0, 0.2]
         R: [1.0, 1.0, 0.1]
         motion_model: "DiffDrive"
         reference_name: "odom"
         path_dist_gain: 1.0
         goal_dist_gain: 1.0
         goal_angle_gain: 0.5
         xy_goal_tolerance: 0.25
         yaw_goal_tolerance: 0.25
         state_bounds_warning: true
         cmd_timeout: 1.0
         velocity_scaling_factor: 1.0
         max_iterations: 3

   local_costmap:
     local_costmap:
       ros__parameters:
         update_frequency: 5.0
         publish_frequency: 2.0
         global_frame: odom
         robot_base_frame: base_link
         use_sim_time: True
         rolling_window: true
         width: 6
         height: 6
         resolution: 0.05
         robot_radius: 0.3
         plugins: ["voxel_layer", "inflation_layer"]
         inflation_layer:
           plugin: "nav2_costmap_2d::InflationLayer"
           cost_scaling_factor: 3.0
           inflation_radius: 0.55
         voxel_layer:
           plugin: "nav2_costmap_2d::VoxelLayer"
           enabled: True
           publish_voxel_map: False
           origin_z: 0.0
           z_resolution: 0.2
           z_voxels: 8
           max_obstacle_height: 2.0
           mark_threshold: 0
           observation_sources: scan
           scan:
             topic: /scan
             max_obstacle_height: 2.0
             clearing: True
             marking: True
             data_type: "LaserScan"
             raytrace_max_range: 3.0
             raytrace_min_range: 0.0
             obstacle_max_range: 2.5
             obstacle_min_range: 0.0
         static_layer:
           plugin: "nav2_costmap_2d::StaticLayer"
           map_subscribe_transient_local: True
         always_send_full_costmap: True

   global_costmap:
     global_costmap:
       ros__parameters:
         update_frequency: 1.0
         publish_frequency: 1.0
         global_frame: map
         robot_base_frame: base_link
         use_sim_time: True
         robot_radius: 0.3
         resolution: 0.05
         track_unknown_space: true
         plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
         obstacle_layer:
           plugin: "nav2_costmap_2d::ObstacleLayer"
           enabled: True
           observation_sources: scan
           scan:
             topic: /scan
             max_obstacle_height: 2.0
             clearing: True
             marking: True
             data_type: "LaserScan"
             raytrace_max_range: 3.0
             raytrace_min_range: 0.0
             obstacle_max_range: 2.5
             obstacle_min_range: 0.0
         static_layer:
           plugin: "nav2_costmap_2d::StaticLayer"
           map_subscribe_transient_local: True
         inflation_layer:
           plugin: "nav2_costmap_2d::InflationLayer"
           cost_scaling_factor: 3.0
           inflation_radius: 0.55
         always_send_full_costmap: True

   planner_server:
     ros__parameters:
       expected_planner_frequency: 20.0
       use_sim_time: True
       planner_plugins: ["GridBased"]
       GridBased:
         plugin: "nav2_navfn_planner::NavfnPlanner"
         tolerance: 0.5
         use_astar: false
         allow_unknown: true
   ```

2. Create a launch file for Nav2 with humanoid-specific parameters:
   ```python
   import os
   from ament_index_python.packages import get_package_share_directory
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.conditions import IfCondition
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node


   def generate_launch_description():
       # Launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time')
       params_file = LaunchConfiguration('params_file')
       autostart = LaunchConfiguration('autostart')
       use_composition = LaunchConfiguration('use_composition')

       # Paths
       bringup_dir = get_package_share_directory('nav2_bringup')
       params_file_default = os.path.join(bringup_dir, 'params', 'nav2_params.yaml')

       # Declare launch arguments
       declare_use_sim_time_cmd = DeclareLaunchArgument(
           'use_sim_time',
           default_value='True',
           description='Use simulation (Gazebo) clock if true')

       declare_params_file_cmd = DeclareLaunchArgument(
           'params_file',
           default_value=params_file_default,
           description='Full path to the ROS2 parameters file to use for all launched nodes')

       declare_autostart_cmd = DeclareLaunchArgument(
           'autostart',
           default_value='True',
           description='Automatically startup the nav2 stack')

       declare_use_composition_cmd = DeclareLaunchArgument(
           'use_composition',
           default_value='False',
           description='Use composed bringup if True')

       # Launch the main Nav2 launch file
       nav2_bringup_launch = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(bringup_dir, 'launch', 'navigation_launch.py')),
           launch_arguments={
               'use_sim_time': use_sim_time,
               'params_file': params_file,
               'autostart': autostart,
               'use_composition': use_composition,
           }.items())

       return LaunchDescription([
           declare_use_sim_time_cmd,
           declare_params_file_cmd,
           declare_autostart_cmd,
           declare_use_composition_cmd,
           nav2_bringup_launch
       ])
   ```

3. Launch Nav2 with your robot in simulation:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 launch nav2_humanoid_launch.py
   ```

## Exercise 6: Integration and Testing

Finally, you'll integrate all components and test the complete AI-robot brain system.

### Steps

1. Create a complete launch file that brings up:
   - Isaac Sim with robot and sensors
   - Isaac ROS VSLAM pipeline
   - Nav2 navigation stack
   - Robot control interface

2. Test the complete pipeline:
   - Verify sensor data flows correctly
   - Check VSLAM localization accuracy
   - Test navigation to specified goals
   - Monitor system performance

## Key Takeaways

- Isaac Sim provides a comprehensive platform for AI development in robotics
- Synthetic data generation can significantly reduce real-world data collection needs
- Isaac ROS packages provide hardware-accelerated perception capabilities
- Nav2 can be configured for specialized robots like humanoid platforms
- Integration requires careful attention to timing and coordinate frames

## Key Concepts

- **Synthetic Data Generation**: Creating labeled training data in simulation
- **Domain Randomization**: Technique to improve real-world transfer of AI models
- **Isaac ROS VSLAM**: Hardware-accelerated visual SLAM pipeline
- **Nav2 Navigation**: ROS 2 navigation stack for mobile robots
- **Photorealistic Rendering**: High-fidelity visualization for perception training
- **Hardware Acceleration**: GPU utilization for real-time AI processing

## Practical Exercises

1. Generate a synthetic dataset for object detection in your specific domain
2. Train a perception model using synthetic data and test on real data
3. Implement a complete navigation pipeline with VSLAM localization
4. Evaluate the performance of domain randomization techniques

## Common Failure Modes

- **GPU Resource Exhaustion**: Complex simulations consuming excessive GPU memory
- **Domain Gap**: Poor transfer from simulation to real-world due to insufficient domain randomization
- **SLAM Drift**: Accumulation of localization errors over time
- **Perception Failures**: AI models failing in real-world conditions despite synthetic training
- **Performance Bottlenecks**: GPU-accelerated pipelines not providing expected performance gains
- **Coordinate System Mismatches**: Different frames causing navigation errors