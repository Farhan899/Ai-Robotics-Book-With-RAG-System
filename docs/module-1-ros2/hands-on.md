# Hands-On: ROS 2 Nodes, Topics, Services, and Actions

## Exercise 1: Creating Your First ROS 2 Node

In this exercise, you'll create a simple ROS 2 node that publishes messages to a topic.

### Prerequisites
- ROS 2 Humble installed
- Basic Python knowledge
- Terminal access

### Steps

1. Create a new Python package for your ROS 2 nodes:
   ```bash
   mkdir -p ~/ros2_ws/src/my_robot_nodes
   cd ~/ros2_ws/src/my_robot_nodes
   ```

2. Create a `setup.py` file:
   ```python
   from setuptools import setup

   package_name = 'my_robot_nodes'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[],
       py_modules=[],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Simple ROS 2 nodes for learning',
       license='Apache License 2.0',
   )
   ```

3. Create a `publisher_node.py` file:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String


   class MinimalPublisher(Node):

       def __init__(self):
           super().__init__('minimal_publisher')
           self.publisher_ = self.create_publisher(String, 'topic', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = 'Hello World: %d' % self.i
           self.publisher_.publish(msg)
           self.get_logger().info('Publishing: "%s"' % msg.data)
           self.i += 1


   def main(args=None):
       rclpy.init(args=args)

       minimal_publisher = MinimalPublisher()

       rclpy.spin(minimal_publisher)

       # Destroy the node explicitly
       minimal_publisher.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

4. Make the file executable:
   ```bash
   chmod +x publisher_node.py
   ```

5. Source your ROS 2 installation and run the publisher:
   ```bash
   source /opt/ros/humble/setup.bash
   cd ~/ros2_ws/src/my_robot_nodes
   python3 publisher_node.py
   ```

6. In a new terminal, listen to the topic:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 topic echo /topic std_msgs/msg/String
   ```

## Exercise 2: Creating a Subscriber Node

Now create a subscriber node that receives messages from the publisher.

1. Create a `subscriber_node.py` file:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String


   class MinimalSubscriber(Node):

       def __init__(self):
           super().__init__('minimal_subscriber')
           self.subscription = self.create_subscription(
               String,
               'topic',
               self.listener_callback,
               10)
           self.subscription  # prevent unused variable warning

       def listener_callback(self, msg):
           self.get_logger().info('I heard: "%s"' % msg.data)


   def main(args=None):
       rclpy.init(args=args)

       minimal_subscriber = MinimalSubscriber()

       rclpy.spin(minimal_subscriber)

       # Destroy the node explicitly
       minimal_subscriber.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. Make the file executable and run it:
   ```bash
   chmod +x subscriber_node.py
   source /opt/ros/humble/setup.bash
   python3 subscriber_node.py
   ```

## Exercise 3: Creating a Service Server and Client

Create a simple service that adds two integers.

### Service Server

1. Create `service_server.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from example_interfaces.srv import AddTwoInts


   class MinimalService(Node):

       def __init__(self):
           super().__init__('minimal_service')
           self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

       def add_two_ints_callback(self, request, response):
           response.sum = request.a + request.b
           self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
           return response


   def main(args=None):
       rclpy.init(args=args)

       minimal_service = MinimalService()

       rclpy.spin(minimal_service)

       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

### Service Client

2. Create `service_client.py`:
   ```python
   import sys
   import rclpy
   from rclpy.node import Node
   from example_interfaces.srv import AddTwoInts


   class MinimalClient(Node):

       def __init__(self):
           super().__init__('minimal_client')
           self.cli = self.create_client(AddTwoInts, 'add_two_ints')
           while not self.cli.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('service not available, waiting again...')
           self.req = AddTwoInts.Request()

       def send_request(self, a, b):
           self.req.a = a
           self.req.b = b
           future = self.cli.call_async(self.req)
           rclpy.spin_until_future_complete(self, future)
           return future.result()


   def main(args=None):
       rclpy.init(args=args)

       minimal_client = MinimalClient()
       response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
       minimal_client.get_logger().info(
           'Result of add_two_ints: for %d + %d = %d' %
           (int(sys.argv[1]), int(sys.argv[2]), response.sum))

       minimal_client.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. Run the service server in one terminal:
   ```bash
   chmod +x service_server.py
   source /opt/ros/humble/setup.bash
   python3 service_server.py
   ```

4. In another terminal, call the service:
   ```bash
   chmod +x service_client.py
   source /opt/ros/humble/setup.bash
   python3 service_client.py 2 3
   ```

## Exercise 4: Creating a Simple URDF

Create a minimal humanoid URDF file.

1. Create a `urdf` directory in your package:
   ```bash
   mkdir urdf
   ```

2. Create `simple_humanoid.urdf`:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_humanoid">
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

     <!-- Head -->
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

## Exercise 5: Visualizing the Robot in RViz

1. Create a launch file `display.launch.py`:
   ```python
   import os
   from ament_index_python.packages import get_package_share_directory
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node


   def generate_launch_description():
       # Find the URDF file
       urdf_file = os.path.join(
           get_package_share_directory('my_robot_nodes'),
           'urdf',
           'simple_humanoid.urdf'
       )

       # Read the URDF file
       with open(urdf_file, 'r') as infp:
           robot_desc = infp.read()

       # Launch configuration
       use_sim_time = LaunchConfiguration('use_sim_time', default='false')

       # Declare launch arguments
       declare_use_sim_time_argument = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation/Gazebo clock')

       # Robot state publisher node
       params = {'use_sim_time': use_sim_time, 'robot_description': robot_desc}
       start_robot_state_publisher_cmd = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           name='robot_state_publisher',
           output='screen',
           parameters=[params]
       )

       # Joint state publisher GUI node
       joint_state_publisher_gui_node = Node(
           package='joint_state_publisher_gui',
           executable='joint_state_publisher_gui',
           name='joint_state_publisher_gui',
           output='screen'
       )

       # RViz node
       rviz_node = Node(
           package='rviz2',
           executable='rviz2',
           name='rviz2',
           output='screen'
       )

       # Create launch description and add actions
       ld = LaunchDescription()

       ld.add_action(declare_use_sim_time_argument)
       ld.add_action(start_robot_state_publisher_cmd)
       ld.add_action(joint_state_publisher_gui_node)
       ld.add_action(rviz_node)

       return ld
   ```

2. Launch the visualization:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 launch display.launch.py
   ```

## Key Takeaways

- Nodes are the basic execution units in ROS 2
- Topics enable asynchronous publish-subscribe communication
- Services provide synchronous request-response communication
- Actions are for long-running tasks with feedback
- URDF describes robot kinematics, dynamics, and visual properties

## Key Concepts

- **Node**: A process that performs computation
- **Topic**: Named bus for message exchange
- **Service**: Synchronous request/response communication
- **Action**: Asynchronous goal-based communication with feedback
- **URDF**: Unified Robot Description Format for robot modeling
- **RViz**: 3D visualization tool for ROS

## Common Failure Modes

- **Node Communication Issues**: Nodes may not communicate if topics/services have different names or incompatible message types
- **Parameter Configuration**: Incorrect QoS settings can prevent communication
- **URDF Parsing Errors**: Syntax errors in URDF files can prevent robot visualization
- **Launch File Issues**: Missing dependencies or incorrect paths in launch files can cause failures