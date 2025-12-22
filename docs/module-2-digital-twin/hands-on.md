# Hands-On: Digital Twin with Gazebo and Sensor Simulation

## Exercise 1: Setting Up Gazebo with ROS 2

In this exercise, you'll set up Gazebo with ROS 2 integration and launch a basic simulation.

### Prerequisites
- ROS 2 Humble installed with Gazebo packages
- Basic understanding of ROS 2 concepts (completed Module 1)

### Steps

1. Verify Gazebo installation:
   ```bash
   # Check if Gazebo is available
   gz --version

   # Check ROS 2 Gazebo packages
   ros2 pkg list | grep gazebo
   ```

2. Source your ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

3. Launch Gazebo with ROS 2 bridge:
   ```bash
   # Launch empty world
   ros2 launch gazebo_ros gazebo.launch.py
   ```

4. In a new terminal, check available topics:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 topic list
   ```

## Exercise 2: Spawning a Robot in Gazebo

Now you'll spawn a robot model in Gazebo using the URDF from Module 1.

### Steps

1. Create a launch file `spawn_robot.launch.py` in your workspace:
   ```python
   import os
   from ament_index_python.packages import get_package_share_directory
   from launch import LaunchDescription
   from launch.actions import IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch_ros.actions import Node


   def generate_launch_description():
       # Gazebo launch
       gazebo = IncludeLaunchDescription(
           PythonLaunchDescriptionSource([os.path.join(
               get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
       )

       # Robot state publisher node
       # (Assuming you have the URDF from Module 1)
       with open(os.path.join(get_package_share_directory('my_robot_nodes'), 'urdf', 'simple_humanoid.urdf'), 'r') as infp:
           robot_desc = infp.read()

       params = {'use_sim_time': True, 'robot_description': robot_desc}
       node_robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           output='screen',
           parameters=[params]
       )

       # Spawn entity node
       spawn_entity = Node(
           package='gazebo_ros',
           executable='spawn_entity.py',
           arguments=['-topic', 'robot_description', '-entity', 'simple_humanoid'],
           output='screen'
       )

       return LaunchDescription([
           gazebo,
           node_robot_state_publisher,
           spawn_entity,
       ])
   ```

2. Create a Gazebo world file `simple_world.world`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="simple_world">
       <physics type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1</real_time_factor>
         <real_time_update_rate>1000</real_time_update_rate>
       </physics>

       <light name="sun" type="directional">
         <cast_shadows>true</cast_shadows>
         <pose>0 0 10 0 0 0</pose>
         <diffuse>0.8 0.8 0.8 1</diffuse>
         <specular>0.2 0.2 0.2 1</specular>
         <attenuation>
           <range>1000</range>
           <constant>0.9</constant>
           <linear>0.01</linear>
           <quadratic>0.001</quadratic>
         </attenuation>
         <direction>-0.4 0.2 -0.9</direction>
       </light>

       <model name="ground_plane">
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
             <material>
               <ambient>0.3 0.3 0.3 1</ambient>
               <diffuse>0.5 0.5 0.5 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <model name="simple_cube">
         <pose>1 0 0.5 0 0 0</pose>
         <link name="link">
           <inertial>
             <mass>1.0</mass>
             <inertia>
               <ixx>0.166667</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>0.166667</iyy>
               <iyz>0</iyz>
               <izz>0.166667</izz>
             </inertia>
           </inertial>
           <collision name="collision">
             <geometry>
               <box>
                 <size>1 1 1</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>1 1 1</size>
               </box>
             </geometry>
             <material>
               <ambient>1 0 0 1</ambient>
               <diffuse>1 0 0 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>
     </world>
   </sdf>
   ```

3. Launch the simulation with your robot:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 launch spawn_robot.launch.py
   ```

## Exercise 3: Attaching Simulated Sensors

Now you'll add sensors to your robot model and verify they work in simulation.

### Steps

1. Update your URDF file to include a camera sensor:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_humanoid_with_sensors">
     <!-- Base link -->
     <link name="base_link">
       <visual>
         <geometry>
           <box size="0.5 0.2 0.2"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.5 0.2 0.2"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
       </inertial>
     </link>

     <!-- Head with camera -->
     <link name="head">
       <visual>
         <geometry>
           <sphere radius="0.1"/>
         </geometry>
         <material name="white">
           <color rgba="1 1 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <sphere radius="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.5"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <joint name="neck_joint" type="fixed">
       <parent link="base_link"/>
       <child link="head"/>
       <origin xyz="0.3 0 0"/>
     </joint>

     <!-- Camera sensor -->
     <gazebo reference="head">
       <sensor name="camera_sensor" type="camera">
         <always_on>true</always_on>
         <visualize>true</visualize>
         <camera name="head_camera">
           <horizontal_fov>1.047</horizontal_fov>
           <image>
             <width>640</width>
             <height>480</height>
             <format>R8G8B8</format>
           </image>
           <clip>
             <near>0.1</near>
             <far>100</far>
           </clip>
         </camera>
         <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
           <frame_name>head</frame_name>
           <topic_name>camera/image_raw</topic_name>
         </plugin>
       </sensor>
     </gazebo>

     <!-- Left Arm -->
     <link name="left_upper_arm">
       <visual>
         <geometry>
           <cylinder length="0.3" radius="0.05"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.3" radius="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.3"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <joint name="left_shoulder_joint" type="revolute">
       <parent link="base_link"/>
       <child link="left_upper_arm"/>
       <origin xyz="0.1 0.15 0" rpy="0 0 0"/>
       <axis xyz="0 0 1"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
     </joint>
   </robot>
   ```

2. Add a LIDAR sensor to your robot:
   ```xml
   <!-- LIDAR sensor -->
   <gazebo reference="base_link">
     <sensor name="lidar_sensor" type="ray">
       <always_on>true</always_on>
       <visualize>true</visualize>
       <ray>
         <scan>
           <horizontal>
             <samples>640</samples>
             <resolution>1</resolution>
             <min_angle>-1.570796</min_angle>
             <max_angle>1.570796</max_angle>
           </horizontal>
         </scan>
         <range>
           <min>0.1</min>
           <max>10.0</max>
           <resolution>0.01</resolution>
         </range>
       </ray>
       <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
         <ros>
           <namespace>laser</namespace>
           <remapping>~/out:=scan</remapping>
         </ros>
         <output_type>sensor_msgs/LaserScan</output_type>
       </plugin>
     </sensor>
   </gazebo>
   ```

3. Add an IMU sensor to your robot:
   ```xml
   <!-- IMU sensor -->
   <gazebo reference="base_link">
     <sensor name="imu_sensor" type="imu">
       <always_on>true</always_on>
       <visualize>false</visualize>
       <update_rate>100</update_rate>
       <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
         <ros>
           <namespace>imu</namespace>
           <remapping>~/out:=data</remapping>
         </ros>
         <frame_name>base_link</frame_name>
       </plugin>
     </sensor>
   </gazebo>
   ```

## Exercise 4: Streaming Sensor Data to ROS 2

Now you'll verify that sensor data is being published to ROS 2 topics.

### Steps

1. Launch your robot with sensors in Gazebo (using the updated URDF):
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 launch spawn_robot.launch.py
   ```

2. In a new terminal, check available topics:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 topic list
   ```

3. View camera images:
   ```bash
   ros2 run image_view image_view __ns:=/camera
   ```

4. Echo LIDAR data:
   ```bash
   ros2 topic echo /scan sensor_msgs/msg/LaserScan
   ```

5. Echo IMU data:
   ```bash
   ros2 topic echo /imu/data sensor_msgs/msg/Imu
   ```

## Exercise 5: Unity-ROS Bridge (Conceptual)

This exercise demonstrates the conceptual approach to Unity-ROS integration:

### Unity Setup
1. Install Unity-ROS-TCP-Connector package in Unity
2. Configure ROS TCP endpoint to connect to your ROS 2 bridge
3. Set up Unity scene with robot model and sensors

### ROS Bridge
1. Use rosbridge_suite to create a WebSocket connection
2. Implement message serialization for Unity communication
3. Synchronize Unity visualization with Gazebo physics

## Key Takeaways

- Gazebo integrates seamlessly with ROS 2 through gazebo_ros packages
- Robot models can be enhanced with simulated sensors using Gazebo plugins
- Sensor data streams to ROS 2 topics for processing and visualization
- Multiple physics engines offer different trade-offs for simulation quality and performance

## Key Concepts

- **Gazebo Plugins**: Extensions that provide ROS 2 interfaces for Gazebo simulation
- **Sensor Simulation**: Virtual sensors that generate realistic sensor data
- **Physics Engines**: Software components that simulate physical interactions (ODE, Bullet, DART)
- **Spawn Entity**: Process of adding a robot model to a running Gazebo simulation
- **Gazebo World**: SDF file defining the simulation environment
- **ROS-TCP Bridge**: Connection between Unity and ROS for visualization

## Practical Exercises

1. Create a simulation environment with multiple objects and obstacles
2. Implement a mobile robot with differential drive in Gazebo
3. Add multiple sensor types to your robot (camera, LIDAR, IMU)
4. Create a Unity scene that visualizes the same robot state

## Common Failure Modes

- **Model Spawning Issues**: Incorrect URDF format or missing dependencies preventing robot spawn
- **Sensor Connection Problems**: Sensor topics not connecting properly to ROS 2
- **Physics Instability**: Incorrect physical parameters causing simulation artifacts
- **Performance Issues**: Complex models or sensors causing slow simulation
- **Coordinate System Mismatches**: Different coordinate systems between tools causing alignment issues