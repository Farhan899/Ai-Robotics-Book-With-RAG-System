# Hands-On: Vision-Language-Action Implementation

## Exercise 1: Setting Up Voice Command Ingestion System

In this exercise, you'll set up a complete voice command ingestion system that converts speech to text and prepares it for processing.

### Prerequisites
- ROS 2 Humble installed
- Microphone connected to your system
- Python 3.8+ with required packages

### Steps

1. Install required packages for speech recognition:
   ```bash
   pip3 install speechrecognition pyaudio vosk transformers torch
   # For OpenAI Whisper (optional):
   pip3 install openai-whisper
   ```

2. Create a ROS 2 package for VLA components:
   ```bash
   mkdir -p ~/vla_ws/src
   cd ~/vla_ws/src
   ros2 pkg create --build-type ament_python vla_voice_ingestion
   cd vla_voice_ingestion
   ```

3. Create the voice ingestion node `voice_ingestion_node.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   import speech_recognition as sr
   import threading
   import queue
   from std_msgs.msg import String
   from audio_common_msgs.msg import AudioData


   class VoiceIngestionNode(Node):

       def __init__(self):
           super().__init__('voice_ingestion_node')

           # Publishers
           self.command_pub = self.create_publisher(String, 'voice_commands', 10)

           # Parameters
           self.silence_threshold = self.declare_parameter('silence_threshold', 500).value
           self.phrase_time_limit = self.declare_parameter('phrase_time_limit', 5.0).value

           # Speech recognition setup
           self.recognizer = sr.Recognizer()
           self.microphone = sr.Microphone()

           # Adjust for ambient noise
           with self.microphone as source:
               self.recognizer.adjust_for_ambient_noise(source)

           # Start voice processing thread
           self.audio_queue = queue.Queue()
           self.listening_thread = threading.Thread(target=self.listen_continuously)
           self.listening_thread.daemon = True
           self.listening_thread.start()

           self.get_logger().info('Voice ingestion node initialized')

       def listen_continuously(self):
           """Continuously listen for voice commands"""
           with self.microphone as source:
               self.get_logger().info("Listening for voice commands...")
               while rclpy.ok():
                   try:
                       # Listen for audio with timeout
                       audio = self.recognizer.listen(
                           source,
                           timeout=1.0,
                           phrase_time_limit=self.phrase_time_limit
                       )

                       # Add audio to queue for processing
                       self.audio_queue.put(audio)

                   except sr.WaitTimeoutError:
                       # Continue listening
                       continue
                   except Exception as e:
                       self.get_logger().error(f'Error in listening: {e}')
                       continue

       def process_audio(self):
           """Process audio from queue"""
           try:
               while not self.audio_queue.empty():
                   audio = self.audio_queue.get_nowait()

                   # Try to recognize speech using Google (online)
                   try:
                       text = self.recognizer.recognize_google(audio)
                       self.get_logger().info(f'Recognized: {text}')

                       # Publish recognized command
                       cmd_msg = String()
                       cmd_msg.data = text
                       self.command_pub.publish(cmd_msg)

                   except sr.UnknownValueError:
                       self.get_logger().info('Could not understand audio')
                   except sr.RequestError as e:
                       self.get_logger().error(f'Could not request results: {e}')

           except queue.Empty:
               pass  # No audio in queue


   def main(args=None):
       rclpy.init(args=args)
       node = VoiceIngestionNode()

       try:
           while rclpy.ok():
               node.process_audio()
               rclpy.spin_once(node, timeout_sec=0.1)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

4. Make the node executable and test it:
   ```bash
   chmod +x voice_ingestion_node.py
   cd ~/vla_ws
   colcon build --packages-select vla_voice_ingestion
   source install/setup.bash
   ros2 run vla_voice_ingestion voice_ingestion_node.py
   ```

5. In another terminal, listen to recognized commands:
   ```bash
   ros2 topic echo /voice_commands std_msgs/msg/String
   ```

## Exercise 2: Implementing Natural Language Command Parsing

Now you'll implement a natural language command parser that converts text commands into structured representations.

### Steps

1. Create a command parsing node `command_parser_node.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import Pose
   from vla_interfaces.msg import ParsedCommand  # You'll need to create this message type
   import re
   import json


   class CommandParserNode(Node):

       def __init__(self):
           super().__init__('command_parser_node')

           # Publishers and Subscribers
           self.command_sub = self.create_subscription(
               String, 'voice_commands', self.command_callback, 10)
           self.parsed_pub = self.create_publisher(
               ParsedCommand, 'parsed_commands', 10)

           # Define command patterns
           self.command_patterns = {
               'move': [
                   r'move to (.+)',
                   r'go to (.+)',
                   r'navigate to (.+)',
                   r'go (.+)',
                   r'move (.+)'
               ],
               'grasp': [
                   r'pick up (.+)',
                   r'grasp (.+)',
                   r'pick (.+)',
                   r'get (.+)',
                   r'grab (.+)'
               ],
               'place': [
                   r'place (.+) at (.+)',
                   r'put (.+) at (.+)',
                   r'place (.+) on (.+)',
                   r'put (.+) on (.+)'
               ],
               'find': [
                   r'find (.+)',
                   r'locate (.+)',
                   r'where is (.+)',
                   r'search for (.+)'
               ]
           }

           # Define location keywords
           self.location_keywords = {
               'kitchen': [0.0, 0.0, 0.0],  # Replace with actual coordinates
               'living room': [2.0, 0.0, 0.0],
               'bedroom': [0.0, 2.0, 0.0],
               'table': [1.0, 1.0, 0.0],
               'couch': [2.0, 1.0, 0.0]
           }

           # Define object keywords
           self.object_keywords = [
               'cup', 'bottle', 'book', 'phone', 'keys', 'apple',
               'banana', 'box', 'toy', 'remote'
           ]

           self.get_logger().info('Command parser node initialized')

       def command_callback(self, msg):
           """Parse incoming voice commands"""
           command_text = msg.data.lower().strip()
           self.get_logger().info(f'Parsing command: {command_text}')

           # Parse the command
           parsed_cmd = self.parse_command(command_text)

           if parsed_cmd:
               # Publish parsed command
               self.parsed_pub.publish(parsed_cmd)
               self.get_logger().info(f'Parsed command: {parsed_cmd.action} {parsed_cmd.parameters}')
           else:
               self.get_logger().warn(f'Could not parse command: {command_text}')

       def parse_command(self, command):
           """Parse natural language command into structured format"""
           # Try each command type
           for action_type, patterns in self.command_patterns.items():
               for pattern in patterns:
                   match = re.search(pattern, command)
                   if match:
                       params = list(match.groups())

                       # Create parsed command message
                       parsed_cmd = ParsedCommand()
                       parsed_cmd.action = action_type
                       parsed_cmd.raw_command = command

                       # Process parameters based on action type
                       if action_type == 'move':
                           parsed_cmd.target_location = self.extract_location(params[0])
                       elif action_type == 'grasp':
                           parsed_cmd.target_object = self.extract_object(params[0])
                       elif action_type == 'place':
                           parsed_cmd.target_object = self.extract_object(params[0])
                           parsed_cmd.target_location = self.extract_location(params[1])
                       elif action_type == 'find':
                           parsed_cmd.target_object = self.extract_object(params[0])

                       return parsed_cmd

           return None  # Command not recognized

       def extract_location(self, location_text):
           """Extract location from text"""
           # Check for predefined locations
           for loc_name, coords in self.location_keywords.items():
               if loc_name in location_text:
                   pose = Pose()
                   pose.position.x = float(coords[0])
                   pose.position.y = float(coords[1])
                   pose.position.z = float(coords[2])
                   return pose

           # If not found, return default position
           pose = Pose()
           pose.position.x = 0.0
           pose.position.y = 0.0
           pose.position.z = 0.0
           return pose

       def extract_object(self, object_text):
           """Extract object from text"""
           # Find the most likely object
           for obj in self.object_keywords:
               if obj in object_text:
                   return obj

           # Return the original text if no known object found
           return object_text


   def main(args=None):
       rclpy.init(args=args)
       node = CommandParserNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. Create a simple message definition for parsed commands (in `msg/ParsedCommand.msg`):
   ```
   string action
   string raw_command
   string target_object
   geometry_msgs/Pose target_location
   string[] parameters
   ```

## Exercise 3: Creating a Task Decomposition System

Now you'll implement a task decomposition system that breaks complex commands into executable steps.

### Steps

1. Create the task decomposition node `task_decomposer_node.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from vla_interfaces.msg import ParsedCommand, TaskPlan
   from geometry_msgs.msg import Pose
   from std_msgs.msg import String


   class TaskDecomposerNode(Node):

       def __init__(self):
           super().__init__('task_decomposer_node')

           # Publishers and Subscribers
           self.parsed_sub = self.create_subscription(
               ParsedCommand, 'parsed_commands', self.parsed_callback, 10)
           self.plan_pub = self.create_publisher(TaskPlan, 'task_plans', 10)

           # Define task templates
           self.task_templates = {
               'move_and_grasp': [
                   {'action': 'navigate_to', 'required_params': ['target_location']},
                   {'action': 'detect_object', 'required_params': ['target_object']},
                   {'action': 'approach_object', 'required_params': ['target_object']},
                   {'action': 'grasp_object', 'required_params': ['target_object']}
               ],
               'grasp_and_place': [
                   {'action': 'navigate_to', 'required_params': ['start_location']},
                   {'action': 'detect_object', 'required_params': ['target_object']},
                   {'action': 'grasp_object', 'required_params': ['target_object']},
                   {'action': 'navigate_to', 'required_params': ['target_location']},
                   {'action': 'place_object', 'required_params': ['target_object']}
               ],
               'find_and_report': [
                   {'action': 'detect_object', 'required_params': ['target_object']},
                   {'action': 'report_location', 'required_params': ['target_object']}
               ]
           }

           self.get_logger().info('Task decomposer node initialized')

       def parsed_callback(self, msg):
           """Process parsed commands and generate task plans"""
           self.get_logger().info(f'Decomposing command: {msg.action}')

           # Generate task plan based on command type
           task_plan = self.generate_task_plan(msg)

           if task_plan and len(task_plan.tasks) > 0:
               self.plan_pub.publish(task_plan)
               self.get_logger().info(f'Generated plan with {len(task_plan.tasks)} tasks')
           else:
               self.get_logger().warn('Could not generate task plan')

       def generate_task_plan(self, parsed_cmd):
           """Generate a task plan from parsed command"""
           plan = TaskPlan()
           plan.header.stamp = self.get_clock().now().to_msg()
           plan.header.frame_id = 'map'
           plan.original_command = parsed_cmd.raw_command

           # Determine task template based on action and parameters
           if parsed_cmd.action == 'grasp' and parsed_cmd.target_location.position.x != 0.0:
               # This is likely a grasp and place command
               template = self.task_templates['grasp_and_place']

               # Update parameters for the plan
               for task in template:
                   if 'start_location' in task['required_params']:
                       # For now, assume current location is [0,0,0]
                       task['params'] = {'location': [0.0, 0.0, 0.0]}
                   elif 'target_object' in task['required_params']:
                       task['params'] = {'object': parsed_cmd.target_object}
                   elif 'target_location' in task['required_params']:
                       task['params'] = {
                           'location': [
                               parsed_cmd.target_location.position.x,
                               parsed_cmd.target_location.position.y,
                               parsed_cmd.target_location.position.z
                           ]
                       }
                   else:
                       task['params'] = {}

                   # Add to plan
                   plan.tasks.append(self.create_task_msg(task))
           elif parsed_cmd.action == 'grasp':
               # Simple grasp command
               template = [
                   {'action': 'navigate_to', 'required_params': ['target_object'], 'params': {}},
                   {'action': 'detect_object', 'required_params': ['target_object'], 'params': {'object': parsed_cmd.target_object}},
                   {'action': 'grasp_object', 'required_params': ['target_object'], 'params': {'object': parsed_cmd.target_object}}
               ]

               for task in template:
                   if 'target_object' in task['required_params']:
                       task['params'] = {'object': parsed_cmd.target_object}
                   plan.tasks.append(self.create_task_msg(task))
           elif parsed_cmd.action == 'move':
               # Simple navigation command
               template = [
                   {'action': 'navigate_to', 'required_params': ['target_location'], 'params': {
                       'location': [
                           parsed_cmd.target_location.position.x,
                           parsed_cmd.target_location.position.y,
                           parsed_cmd.target_location.position.z
                       ]
                   }}
               ]

               for task in template:
                   plan.tasks.append(self.create_task_msg(task))
           else:
               # Default: simple action
               template = [{'action': parsed_cmd.action, 'required_params': [], 'params': {}}]
               for task in template:
                   plan.tasks.append(self.create_task_msg(task))

           return plan

       def create_task_msg(self, task_dict):
           """Create a Task message from dictionary"""
           from vla_interfaces.msg import Task  # Assuming you have this message type

           task_msg = Task()
           task_msg.action = task_dict['action']
           task_msg.parameters = json.dumps(task_dict.get('params', {}))
           task_msg.priority = 0  # Default priority

           return task_msg


   def main(args=None):
       rclpy.init(args=args)
       node = TaskDecomposerNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

## Exercise 4: Building Language-to-ROS Action Mapping System

Now you'll create a system that maps language-based task plans to ROS 2 actions and services.

### Steps

1. Create the action mapper node `action_mapper_node.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from vla_interfaces.msg import TaskPlan, ExecutableAction
   from geometry_msgs.msg import Pose
   from nav2_msgs.action import NavigateToPose
   from rclpy.action import ActionClient
   import json


   class ActionMapperNode(Node):

       def __init__(self):
           super().__init__('action_mapper_node')

           # Publishers and Subscribers
           self.plan_sub = self.create_subscription(
               TaskPlan, 'task_plans', self.plan_callback, 10)
           self.action_pub = self.create_publisher(
               ExecutableAction, 'executable_actions', 10)

           # Action clients
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

           # Define action mappings
           self.action_mappings = {
               'navigate_to': self.map_navigate_to,
               'grasp_object': self.map_grasp_object,
               'place_object': self.map_place_object,
               'detect_object': self.map_detect_object,
               'approach_object': self.map_approach_object,
               'report_location': self.map_report_location
           }

           self.get_logger().info('Action mapper node initialized')

       def plan_callback(self, msg):
           """Process task plans and map to executable actions"""
           self.get_logger().info(f'Processing plan with {len(msg.tasks)} tasks')

           for task in msg.tasks:
               self.get_logger().info(f'Mapping task: {task.action}')

               # Map the task to an executable action
               executable_action = self.map_task_to_action(task, msg.header.frame_id)

               if executable_action:
                   self.action_pub.publish(executable_action)
                   self.get_logger().info(f'Published action: {executable_action.action_type}')
               else:
                   self.get_logger().warn(f'Could not map task: {task.action}')

       def map_task_to_action(self, task, frame_id):
           """Map a task to an executable ROS action"""
           if task.action in self.action_mappings:
               return self.action_mappings[task.action](task, frame_id)
           else:
               self.get_logger().warn(f'Unknown action: {task.action}')
               return None

       def map_navigate_to(self, task, frame_id):
           """Map navigate_to task to ROS action"""
           from vla_interfaces.msg import ExecutableAction

           action_msg = ExecutableAction()
           action_msg.action_type = 'navigate_to_pose'
           action_msg.header.stamp = self.get_clock().now().to_msg()
           action_msg.header.frame_id = frame_id

           # Parse parameters
           params = json.loads(task.parameters) if task.parameters else {}

           if 'location' in params:
               location = params['location']

               # Create NavigateToPose goal
               goal = NavigateToPose.Goal()
               goal.pose.header.frame_id = frame_id
               goal.pose.pose.position.x = float(location[0])
               goal.pose.pose.position.y = float(location[1])
               goal.pose.pose.position.z = float(location[2])
               # Set orientation to face forward (for simplicity)
               goal.pose.pose.orientation.w = 1.0

               action_msg.goal = json.dumps({
                   'target_pose': {
                       'position': location,
                       'orientation': [0, 0, 0, 1]  # w=1 for no rotation
                   }
               })
           else:
               self.get_logger().warn('Navigate task missing location parameter')
               return None

           return action_msg

       def map_grasp_object(self, task, frame_id):
           """Map grasp_object task to ROS action"""
           from vla_interfaces.msg import ExecutableAction

           action_msg = ExecutableAction()
           action_msg.action_type = 'grasp_object'
           action_msg.header.stamp = self.get_clock().now().to_msg()
           action_msg.header.frame_id = frame_id

           # Parse parameters
           params = json.loads(task.parameters) if task.parameters else {}

           if 'object' in params:
               action_msg.goal = json.dumps({'object_name': params['object']})
           else:
               self.get_logger().warn('Grasp task missing object parameter')
               return None

           return action_msg

       def map_place_object(self, task, frame_id):
           """Map place_object task to ROS action"""
           from vla_interfaces.msg import ExecutableAction

           action_msg = ExecutableAction()
           action_msg.action_type = 'place_object'
           action_msg.header.stamp = self.get_clock().now().to_msg()
           action_msg.header.frame_id = frame_id

           # Parse parameters
           params = json.loads(task.parameters) if task.parameters else {}

           if 'object' in params:
               action_msg.goal = json.dumps({'object_name': params['object']})
           else:
               self.get_logger().warn('Place task missing object parameter')
               return None

           return action_msg

       def map_detect_object(self, task, frame_id):
           """Map detect_object task to ROS action"""
           from vla_interfaces.msg import ExecutableAction

           action_msg = ExecutableAction()
           action_msg.action_type = 'detect_object'
           action_msg.header.stamp = self.get_clock().now().to_msg()
           action_msg.header.frame_id = frame_id

           # Parse parameters
           params = json.loads(task.parameters) if task.parameters else {}

           if 'object' in params:
               action_msg.goal = json.dumps({'object_name': params['object']})
           else:
               self.get_logger().warn('Detect task missing object parameter')
               return None

           return action_msg

       def map_approach_object(self, task, frame_id):
           """Map approach_object task to ROS action"""
           from vla_interfaces.msg import ExecutableAction

           action_msg = ExecutableAction()
           action_msg.action_type = 'approach_object'
           action_msg.header.stamp = self.get_clock().now().to_msg()
           action_msg.header.frame_id = frame_id

           # Parse parameters
           params = json.loads(task.parameters) if task.parameters else {}

           if 'object' in params:
               action_msg.goal = json.dumps({'object_name': params['object']})
           else:
               self.get_logger().warn('Approach task missing object parameter')
               return None

           return action_msg

       def map_report_location(self, task, frame_id):
           """Map report_location task to ROS action"""
           from vla_interfaces.msg import ExecutableAction

           action_msg = ExecutableAction()
           action_msg.action_type = 'report_location'
           action_msg.header.stamp = self.get_clock().now().to_msg()
           action_msg.header.frame_id = frame_id

           # Parse parameters
           params = json.loads(task.parameters) if task.parameters else {}

           if 'object' in params:
               action_msg.goal = json.dumps({'object_name': params['object']})
           else:
               self.get_logger().warn('Report task missing object parameter')
               return None

           return action_msg


   def main(args=None):
       rclpy.init(args=args)
       node = ActionMapperNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

## Exercise 5: Complete VLA Integration

Finally, you'll create a launch file that brings together all the VLA components.

### Steps

1. Create a launch file `vla_system_launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, RegisterEventHandler
   from launch.event_handlers import OnProcessStart
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time')

       # Voice ingestion node
       voice_ingestion_node = Node(
           package='vla_voice_ingestion',
           executable='voice_ingestion_node.py',
           name='voice_ingestion_node',
           parameters=[{'use_sim_time': use_sim_time}],
           output='screen'
       )

       # Command parser node
       command_parser_node = Node(
           package='vla_command_parsing',
           executable='command_parser_node.py',
           name='command_parser_node',
           parameters=[{'use_sim_time': use_sim_time}],
           output='screen'
       )

       # Task decomposer node
       task_decomposer_node = Node(
           package='vla_task_decomposition',
           executable='task_decomposer_node.py',
           name='task_decomposer_node',
           parameters=[{'use_sim_time': use_sim_time}],
           output='screen'
       )

       # Action mapper node
       action_mapper_node = Node(
           package='vla_action_mapping',
           executable='action_mapper_node.py',
           name='action_mapper_node',
           parameters=[{'use_sim_time': use_sim_time}],
           output='screen'
       )

       # Action executor node (placeholder - you would implement this)
       action_executor_node = Node(
           package='vla_action_execution',
           executable='action_executor_node.py',
           name='action_executor_node',
           parameters=[{'use_sim_time': use_sim_time}],
           output='screen'
       )

       return LaunchDescription([
           voice_ingestion_node,
           command_parser_node,
           task_decomposer_node,
           action_mapper_node,
           action_executor_node
       ])
   ```

2. Launch the complete VLA system:
   ```bash
   ros2 launch vla_system_launch.py
   ```

3. Test the system by speaking commands to your microphone and observing the processing pipeline.

## Key Takeaways

- Voice command ingestion requires real-time audio processing and ASR systems
- Natural language parsing converts text commands to structured representations
- Task decomposition breaks complex commands into executable steps
- Action mapping connects language-based tasks to ROS 2 actions and services
- The complete VLA system integrates all components for natural human-robot interaction

## Key Concepts

- **Automatic Speech Recognition (ASR)**: Converting speech to text
- **Natural Language Understanding (NLU)**: Interpreting the meaning of text
- **Task Decomposition**: Breaking high-level commands into primitive actions
- **Action Mapping**: Connecting language concepts to robot capabilities
- **Real-time Processing**: Handling voice input with low latency
- **Command Parsing**: Extracting intent and entities from natural language
- **Executable Actions**: ROS 2 actions and services that implement robot behaviors

## Practical Exercises

1. Implement a VLA system for your specific robot platform
2. Add support for more complex commands and interactions
3. Integrate visual feedback with voice commands
4. Implement error recovery and human-in-the-loop corrections

## Common Failure Modes

- **Audio Quality Issues**: Background noise affecting speech recognition accuracy
- **Command Ambiguity**: Natural language commands not clearly resolved to actions
- **Timing Issues**: Delays in processing affecting natural interaction
- **Context Confusion**: System failing to maintain context across command sequences
- **Resource Limitations**: Computational constraints affecting real-time performance
- **Misinterpretation**: Language commands mapped to incorrect robot actions