# Code Examples: Sensor Data Streaming and Simulation

## Gazebo Sensor Integration Examples

### Camera Sensor Integration

```xml
<!-- Camera sensor definition in URDF/SDF -->
<gazebo reference="camera_link">
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
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>camera/image_raw</topic_name>
      <hack_baseline>0.07</hack_baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### LIDAR Sensor Integration

```xml
<!-- LIDAR sensor definition in URDF/SDF -->
<gazebo reference="lidar_link">
  <sensor name="lidar_sensor" type="ray">
    <always_on>true</always_on>
    <visualize>true</visualize>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Sensor Integration

```xml
<!-- IMU sensor definition in URDF/SDF -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <visualize>false</visualize>
    <update_rate>100</update_rate>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>imu_link</body_name>
      <topic_name>data</topic_name>
      <gaussian_noise>0.00034</gaussian_noise>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

## ROS 2 Sensor Data Processing

### Camera Data Subscriber

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image (example: display it)
        cv2.imshow('Camera View', cv_image)
        cv2.waitKey(1)

        # Log information
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')


def main(args=None):
    rclpy.init(args=args)

    camera_subscriber = CameraSubscriber()

    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        camera_subscriber.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

### LIDAR Data Subscriber

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Extract range data
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges (inf, nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            max_distance = np.max(valid_ranges)
            avg_distance = np.mean(valid_ranges)

            # Log information
            self.get_logger().info(
                f'LIDAR: Min={min_distance:.2f}, Avg={avg_distance:.2f}, Max={max_distance:.2f}'
            )

            # Example: Check for obstacles within 1 meter
            obstacles = valid_ranges[valid_ranges < 1.0]
            if len(obstacles) > 0:
                self.get_logger().info(f'Obstacles detected: {len(obstacles)} points within 1m')
        else:
            self.get_logger().info('No valid LIDAR data')


def main(args=None):
    rclpy.init(args=args)

    lidar_subscriber = LidarSubscriber()

    try:
        rclpy.spin(lidar_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### IMU Data Subscriber

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import math


class ImuSubscriber(Node):

    def __init__(self):
        super().__init__('imu_subscriber')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Extract orientation (quaternion)
        orientation = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        # Extract angular velocity
        angular_velocity = msg.angular_velocity

        # Extract linear acceleration
        linear_acceleration = msg.linear_acceleration

        # Log information
        self.get_logger().info(
            f'IMU: R={roll:.2f}, P={pitch:.2f}, Y={yaw:.2f} | '
            f'Ang Vel: x={angular_velocity.x:.2f}, y={angular_velocity.y:.2f}, z={angular_velocity.z:.2f} | '
            f'Lin Acc: x={linear_acceleration.x:.2f}, y={linear_acceleration.y:.2f}, z={linear_acceleration.z:.2f}'
        )

    def quaternion_to_euler(self, x, y, z, w):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)

    imu_subscriber = ImuSubscriber()

    try:
        rclpy.spin(imu_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        imu_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Gazebo World File Examples

### Simple World with Physics Configuration

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Sun light -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Ground plane -->
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

    <!-- Example objects -->
    <model name="box1">
      <pose>2 0 0.5 0 0 0</pose>
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

    <model name="cylinder1">
      <pose>-2 0 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.291667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.291667</iyy>
            <iyz>0</iyz>
            <izz>0.125</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Unity-ROS Bridge Integration Example

### Unity C# Script for ROS Communication

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosSharp.RosBridgeClient;

public class RobotController : MonoBehaviour
{
    private RosSocket rosSocket;
    private string robotName = "my_robot";

    // Start is called before the first frame update
    void Start()
    {
        // Initialize ROS connection
        RosBridgeClient.RosConnector rosConnector = GetComponent<RosBridgeClient.RosConnector>();
        rosSocket = rosConnector.RosSocket;

        // Subscribe to robot state topic
        rosSocket.Subscribe<Unity.RobotStateMsg>(
            "/robot_state",
            ReceiveRobotState,
            10
        );
    }

    void ReceiveRobotState(Unity.RobotStateMsg message)
    {
        // Update robot position in Unity based on ROS message
        Vector3 newPosition = new Vector3(
            message.position.x,
            message.position.y,
            message.position.z
        );

        transform.position = newPosition;

        // Update robot rotation
        Quaternion newRotation = new Quaternion(
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w
        );

        transform.rotation = newRotation;
    }

    // Update is called once per frame
    void Update()
    {
        // Send robot state to ROS
        if (rosSocket != null)
        {
            Unity.RobotStateMsg robotState = new Unity.RobotStateMsg();
            robotState.position = new Unity.Vector3Msg();
            robotState.position.x = transform.position.x;
            robotState.position.y = transform.position.y;
            robotState.position.z = transform.position.z;

            robotState.orientation = new Unity.QuaternionMsg();
            robotState.orientation.x = transform.rotation.x;
            robotState.orientation.y = transform.rotation.y;
            robotState.orientation.z = transform.rotation.z;
            robotState.orientation.w = transform.rotation.w;

            rosSocket.Publish("/unity_robot_state", robotState);
        }
    }
}
```

## Launch Files for Simulation

### Robot with Sensors Launch File

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch'),
            '/gazebo.launch.py'
        ]),
    )

    # Robot state publisher
    with open(os.path.join(get_package_share_directory('my_robot_nodes'), 'urdf', 'robot_with_sensors.urdf'), 'r') as infp:
        robot_desc = infp.read()

    params = {'use_sim_time': True, 'robot_description': robot_desc}
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # Spawn entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )

    # Sensor processing nodes
    camera_subscriber = Node(
        package='my_robot_nodes',
        executable='camera_subscriber',
        name='camera_subscriber',
        output='screen'
    )

    lidar_subscriber = Node(
        package='my_robot_nodes',
        executable='lidar_subscriber',
        name='lidar_subscriber',
        output='screen'
    )

    imu_subscriber = Node(
        package='my_robot_nodes',
        executable='imu_subscriber',
        name='imu_subscriber',
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        camera_subscriber,
        lidar_subscriber,
        imu_subscriber,
    ])
```

## Key Concepts

- **Gazebo Plugins**: Extensions that provide ROS 2 interfaces for Gazebo simulation
- **Sensor Simulation**: Virtual sensors that generate realistic sensor data
- **Data Streaming**: Real-time transmission of sensor data from simulation to ROS 2
- **Coordinate Systems**: Proper alignment between simulation, robot, and sensor frames
- **Message Types**: Standard ROS 2 message types for different sensor data
- **Noise Models**: Realistic noise addition to simulated sensor data
- **Physics Parameters**: Configuration of physical properties for realistic simulation

## Practical Exercises

1. Create a complete robot model with multiple sensors (camera, LIDAR, IMU)
2. Implement sensor data processing nodes in ROS 2
3. Create a launch file that starts the complete simulation
4. Build a Unity scene that visualizes the same robot state as Gazebo

## Common Failure Modes

- **Sensor Topic Mismatches**: Incorrect topic names preventing data flow
- **Coordinate System Errors**: Misaligned sensor frames causing incorrect data interpretation
- **Performance Bottlenecks**: High-resolution sensors causing simulation slowdown
- **Noise Model Issues**: Incorrect noise parameters making simulation unrealistic
- **Plugin Configuration**: Improperly configured Gazebo plugins causing sensor failures
- **Frame Timing**: Synchronization issues between simulation and visualization