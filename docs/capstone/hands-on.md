---
sidebar_position: 3
---

# Capstone Hands-On: Building the Autonomous Humanoid

## Overview

In this hands-on module, you'll implement the complete autonomous humanoid system by integrating all the components from the previous modules. This comprehensive exercise will demonstrate how to combine ROS 2 middleware, Gazebo simulation, NVIDIA Isaac perception, and Vision-Language-Action capabilities into a functioning voice-controlled robot.

## Prerequisites

Before starting this hands-on exercise, ensure you have:

1. Completed all previous modules (Modules 1-4)
2. All required software installed (ROS 2 Humble, Gazebo, Isaac Sim, Python 3.8+)
3. OpenAI API key configured for LLM integration
4. Working knowledge of ROS 2 concepts (nodes, topics, services, actions)
5. Basic understanding of computer vision and perception systems

## Project Setup

### 1. Create the Project Structure

First, let's set up the workspace for our autonomous humanoid project:

```bash
# Create the workspace
mkdir -p ~/autonomous_humanoid_ws/src
cd ~/autonomous_humanoid_ws

# Create the package structure
cd src
git clone https://github.com/your-organization/humanoid_description.git
git clone https://github.com/your-organization/humanoid_bringup.git
git clone https://github.com/your-organization/humanoid_voice_control.git
git clone https://github.com/your-organization/humanoid_perception.git
git clone https://github.com/your-organization/humanoid_navigation.git
git clone https://github.com/your-organization/humanoid_manipulation.git
git clone https://github.com/your-organization/humanoid_planning.git
git clone https://github.com/your-organization/humanoid_simulation.git
git clone https://github.com/your-organization/humanoid_msgs.git

# Build the workspace
cd ~/autonomous_humanoid_ws
colcon build --packages-select humanoid_msgs
source install/setup.bash
colcon build
source install/setup.bash
```

### 2. Environment Configuration

Set up the necessary environment variables:

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="your-openai-api-key"
export HUMANOID_ROBOT=1
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/autonomous_humanoid_ws/src/humanoid_description/models
export ROS_DOMAIN_ID=42
```

## Voice Command Integration

### 3. Implement Voice Command Processing Node

Create the voice command processing node that will handle speech-to-text conversion:

```python
#!/usr/bin/env python3
# File: humanoid_voice_control/humanoid_voice_control/voice_command_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import speech_recognition as sr
import openai
import json
import threading
import queue
from humanoid_msgs.msg import ParsedCommand
from humanoid_msgs.srv import CommandValidation

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True

        # Audio queue for processing
        self.audio_queue = queue.Queue()

        # Publishers and subscribers
        self.voice_cmd_pub = self.create_publisher(String, '/voice_commands', 10)
        self.parsed_cmd_pub = self.create_publisher(ParsedCommand, '/parsed_commands', 10)
        self.audio_sub = self.create_subscription(
            AudioData, '/audio', self.audio_callback, 10)

        # Service for command validation
        self.validation_srv = self.create_service(
            CommandValidation, '/validate_command', self.validate_command_callback)

        # Timer for audio processing
        self.process_timer = self.create_timer(0.1, self.process_audio)

        # Initialize OpenAI
        openai.api_key = self.get_parameter_or_set('openai_api_key', 'your-openai-key')

        self.get_logger().info('Voice Command Node initialized')

    def get_parameter_or_set(self, name, default_value):
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).get_parameter_value().string_value

    def audio_callback(self, msg):
        """Callback for audio data"""
        self.audio_queue.put(msg)

    def process_audio(self):
        """Process audio from queue"""
        try:
            audio_data = self.audio_queue.get_nowait()
            self.process_audio_data(audio_data)
        except queue.Empty:
            pass

    def process_audio_data(self, audio_data):
        """Process raw audio data to text"""
        try:
            # Convert audio data to audio file format
            with sr.AudioData(audio_data.data, 16000, 2) as source:
                # Perform speech recognition
                text = self.recognizer.recognize_google(source)

                if text:
                    # Publish the recognized text
                    cmd_msg = String()
                    cmd_msg.data = text
                    self.voice_cmd_pub.publish(cmd_msg)

                    # Parse and publish structured command
                    parsed_cmd = self.parse_command_with_llm(text)
                    if parsed_cmd:
                        self.parsed_cmd_pub.publish(parsed_cmd)

        except sr.UnknownValueError:
            self.get_logger().warn('Could not understand audio')
        except sr.RequestError as e:
            self.get_logger().error(f'Could not request results from speech service; {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def parse_command_with_llm(self, text):
        """Use LLM to parse natural language command into structured format"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a command parser for a humanoid robot.
                    Parse the user's natural language command into a structured format.
                    Return JSON with action_type, target_object, target_location, and parameters.
                    Example: {'action_type': 'navigation', 'target_location': 'kitchen',
                    'target_object': null, 'parameters': {}}"""},
                    {"role": "user", "content": text}
                ],
                temperature=0.1
            )

            # Parse the response
            result = json.loads(response.choices[0].message.content)

            # Create parsed command message
            parsed_cmd = ParsedCommand()
            parsed_cmd.action_type = result.get('action_type', '')
            parsed_cmd.target_object = result.get('target_object', '')
            parsed_cmd.target_location = result.get('target_location', '')
            parsed_cmd.parameters = json.dumps(result.get('parameters', {}))
            parsed_cmd.original_command = text

            return parsed_cmd

        except Exception as e:
            self.get_logger().error(f'Error parsing command with LLM: {e}')
            return None

    def validate_command_callback(self, request, response):
        """Validate if a command is safe to execute"""
        # Basic validation logic
        dangerous_keywords = ['harm', 'damage', 'break', 'destroy']

        is_safe = not any(keyword in request.command.lower() for keyword in dangerous_keywords)
        response.is_valid = is_safe
        response.reason = "Command is safe" if is_safe else "Command contains potentially unsafe keywords"

        return response

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down voice command node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 4. Create the Audio Capture Node

Create a node to capture audio from the microphone:

```python
#!/usr/bin/env python3
# File: humanoid_voice_control/humanoid_voice_control/audio_capture_node.py

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioData
import pyaudio
import threading

class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture_node')

        # Publisher for audio data
        self.audio_pub = self.create_publisher(AudioData, '/audio', 10)

        # Audio parameters
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()

        # Start audio capture thread
        self.capture_thread = threading.Thread(target=self.capture_audio)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.get_logger().info('Audio Capture Node initialized')

    def capture_audio(self):
        """Capture audio from microphone"""
        try:
            stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            self.get_logger().info('Audio capture started')

            while rclpy.ok():
                data = stream.read(self.chunk, exception_on_overflow=False)

                # Create and publish audio message
                audio_msg = AudioData()
                audio_msg.data = data
                self.audio_pub.publish(audio_msg)

        except Exception as e:
            self.get_logger().error(f'Error in audio capture: {e}')
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            self.pyaudio.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down audio capture node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 5. Create Launch File for Voice System

Create a launch file to start the voice command system:

```xml
<!-- File: humanoid_voice_control/launch/voice_system.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='humanoid_voice_control',
            executable='audio_capture_node',
            name='audio_capture',
            output='screen'
        ),
        Node(
            package='humanoid_voice_control',
            executable='voice_command_node',
            name='voice_command',
            output='screen',
            parameters=[
                {'openai_api_key': os.environ.get('OPENAI_API_KEY', 'your-key-here')}
            ]
        )
    ])
```

## Perception System Integration

### 6. Implement Perception Node with Isaac ROS

Create a perception node that integrates Isaac ROS for object detection:

```python
#!/usr/bin/env python3
# File: humanoid_perception/humanoid_perception/perception_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from humanoid_msgs.msg import DetectedObject, SemanticScene

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detected_objects', 10)
        self.semantic_pub = self.create_publisher(
            SemanticScene, '/semantic_scene', 10)

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None

        # Object detection parameters
        self.confidence_threshold = 0.5

        self.get_logger().info('Perception Node initialized')

    def camera_info_callback(self, msg):
        """Callback for camera info"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Callback for image data"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection
            detections = self.detect_objects(cv_image)

            # Publish detections
            detection_msg = self.create_detection_message(detections, msg.header)
            self.detection_pub.publish(detection_msg)

            # Create and publish semantic scene
            semantic_msg = self.create_semantic_scene(detections, msg.header)
            self.semantic_pub.publish(semantic_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, image):
        """Perform object detection on image"""
        # This is a simplified example - in practice, you'd use Isaac ROS DNN nodes
        # or integrate with a real object detection model
        detections = []

        # Example: Use a pre-trained model (YOLO, etc.)
        # For this example, we'll use a simple color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect red objects (example)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv, lower_red, upper_red)

        mask_red = mask_red1 + mask_red2

        # Find contours
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)

                detection = {
                    'class': 'red_object',
                    'confidence': 0.8,
                    'bbox': [x, y, w, h],
                    'center': [x + w//2, y + h//2]
                }
                detections.append(detection)

        return detections

    def create_detection_message(self, detections, header):
        """Create vision_msgs/Detection2DArray message"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            if detection['confidence'] > self.confidence_threshold:
                vision_detection = Detection2D()

                # Set bounding box
                vision_detection.bbox.center.x = detection['center'][0]
                vision_detection.bbox.center.y = detection['center'][1]
                vision_detection.bbox.size_x = detection['bbox'][2]
                vision_detection.bbox.size_y = detection['bbox'][3]

                # Set results
                result = ObjectHypothesisWithPose()
                result.hypothesis.class_id = detection['class']
                result.hypothesis.score = detection['confidence']
                vision_detection.results.append(result)

                detection_array.detections.append(vision_detection)

        return detection_array

    def create_semantic_scene(self, detections, header):
        """Create semantic scene message"""
        semantic_scene = SemanticScene()
        semantic_scene.header = header

        for detection in detections:
            if detection['confidence'] > self.confidence_threshold:
                obj = DetectedObject()
                obj.name = detection['class']
                obj.confidence = detection['confidence']
                obj.position.x = detection['center'][0]
                obj.position.y = detection['center'][1]
                # Convert to 3D position if camera matrix is available
                if self.camera_matrix is not None:
                    # Simple depth assumption for example
                    obj.position.z = 1.0

                semantic_scene.objects.append(obj)

        return semantic_scene

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down perception node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Navigation System Integration

### 7. Implement Navigation Node

Create a navigation node that integrates with Nav2:

```python
#!/usr/bin/env python3
# File: humanoid_navigation/humanoid_navigation/navigation_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from humanoid_msgs.msg import ParsedCommand, NavigationGoal
from humanoid_msgs.srv import NavigateToPose
from nav2_msgs.action import NavigateToPose as NavigateToPoseAction
from rclpy.action import ActionClient
import json

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPoseAction, 'navigate_to_pose')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            ParsedCommand, '/parsed_commands', self.command_callback, 10)
        self.nav_goal_pub = self.create_publisher(
            NavigationGoal, '/navigation_goals', 10)

        # Service for navigation
        self.nav_srv = self.create_service(
            NavigateToPose, '/navigate_to_pose', self.navigate_to_pose_callback)

        # Known locations mapping
        self.location_map = {
            'kitchen': Point(x=2.0, y=1.0, z=0.0),
            'living_room': Point(x=-1.0, y=0.0, z=0.0),
            'bedroom': Point(x=0.0, y=-2.0, z=0.0),
            'office': Point(x=1.5, y=-1.0, z=0.0),
            'entrance': Point(x=0.0, y=2.0, z=0.0)
        }

        self.get_logger().info('Navigation Node initialized')

    def command_callback(self, msg):
        """Callback for parsed commands"""
        if msg.action_type == 'navigation':
            self.handle_navigation_command(msg)

    def handle_navigation_command(self, cmd_msg):
        """Handle navigation commands"""
        target_location = cmd_msg.target_location.lower()

        if target_location in self.location_map:
            target_point = self.location_map[target_location]

            # Create navigation goal
            goal_msg = NavigateToPoseAction.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.pose.position = target_point
            goal_msg.pose.pose.orientation.w = 1.0  # No rotation

            # Send navigation goal
            self.send_navigation_goal(goal_msg)

            # Publish navigation goal message
            nav_goal_msg = NavigationGoal()
            nav_goal_msg.target_location = target_location
            nav_goal_msg.target_pose = goal_msg.pose
            self.nav_goal_pub.publish(nav_goal_msg)
        else:
            self.get_logger().warn(f'Unknown location: {target_location}')

    def send_navigation_goal(self, goal_msg):
        """Send navigation goal to Nav2"""
        self.nav_client.wait_for_server()

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')

    def navigate_to_pose_callback(self, request, response):
        """Service callback for navigation"""
        goal_msg = NavigateToPoseAction.Goal()
        goal_msg.pose = request.target_pose

        self.nav_client.wait_for_server()
        send_goal_future = self.nav_client.send_goal_async(goal_msg)

        # For simplicity, return immediately
        # In a real system, you'd wait for completion or implement async handling
        response.success = True
        response.message = "Navigation goal sent"

        return response

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down navigation node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Manipulation System Integration

### 8. Implement Manipulation Node

Create a manipulation node for object interaction:

```python
#!/usr/bin/env python3
# File: humanoid_manipulation/humanoid_manipulation/manipulation_node.py

import rclpy
from rclpy.node import Node
from humanoid_msgs.msg import ParsedCommand, DetectedObject
from humanoid_msgs.srv import ManipulateObject
from geometry_msgs.msg import Pose, Point
import json

class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            ParsedCommand, '/parsed_commands', self.command_callback, 10)
        self.objects_sub = self.create_subscription(
            DetectedObject, '/detected_objects', self.objects_callback, 10)

        # Service for manipulation
        self.manip_srv = self.create_service(
            ManipulateObject, '/manipulate_object', self.manipulate_callback)

        # Detected objects storage
        self.detected_objects = []

        self.get_logger().info('Manipulation Node initialized')

    def command_callback(self, msg):
        """Callback for parsed commands"""
        if msg.action_type == 'manipulation':
            self.handle_manipulation_command(msg)

    def objects_callback(self, msg):
        """Callback for detected objects"""
        self.detected_objects = msg.objects

    def handle_manipulation_command(self, cmd_msg):
        """Handle manipulation commands"""
        target_object = cmd_msg.target_object.lower()

        # Find the object in detected objects
        object_to_manipulate = None
        for obj in self.detected_objects:
            if obj.name.lower() == target_object:
                object_to_manipulate = obj
                break

        if object_to_manipulate:
            # Perform manipulation (simplified for example)
            self.get_logger().info(f'Attempting to manipulate {target_object}')

            # In a real system, you would:
            # 1. Plan grasp trajectory
            # 2. Move to object position
            # 3. Execute grasp
            # 4. Verify success
        else:
            self.get_logger().warn(f'Object {target_object} not found in scene')

    def manipulate_callback(self, request, response):
        """Service callback for manipulation"""
        # Simplified manipulation logic
        try:
            # In a real system, you would implement the full manipulation pipeline
            if request.action == 'pick':
                response.success = self.execute_pick(request.object_name)
            elif request.action == 'place':
                response.success = self.execute_place(request.location)
            elif request.action == 'move':
                response.success = self.execute_move(request.object_name, request.target_pose)
            else:
                response.success = False
                response.message = f'Unknown action: {request.action}'

        except Exception as e:
            response.success = False
            response.message = f'Error during manipulation: {str(e)}'

        if response.success:
            response.message = f'Successfully executed {request.action} action'
        else:
            response.message = f'Failed to execute {request.action} action'

        return response

    def execute_pick(self, object_name):
        """Execute pick action"""
        # Simplified pick logic
        self.get_logger().info(f'Executing pick for {object_name}')
        # In a real system: plan grasp, move arm, close gripper, verify grasp
        return True

    def execute_place(self, location):
        """Execute place action"""
        # Simplified place logic
        self.get_logger().info(f'Executing place at {location}')
        # In a real system: plan placement, move to location, open gripper, verify placement
        return True

    def execute_move(self, object_name, target_pose):
        """Execute move action"""
        # Simplified move logic
        self.get_logger().info(f'Executing move for {object_name} to {target_pose}')
        # In a real system: pick up object, move to target pose, place object
        return True

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down manipulation node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Task Planning Integration

### 9. Implement Task Planning Node

Create a task planning node that coordinates all subsystems:

```python
#!/usr/bin/env python3
# File: humanoid_planning/humanoid_planning/task_planning_node.py

import rclpy
from rclpy.node import Node
from humanoid_msgs.msg import ParsedCommand, TaskPlan, TaskStep
from humanoid_msgs.srv import CommandValidation
from std_msgs.msg import String
import json

class TaskPlanningNode(Node):
    def __init__(self):
        super().__init__('task_planning_node')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            ParsedCommand, '/parsed_commands', self.command_callback, 10)
        self.plan_pub = self.create_publisher(TaskPlan, '/task_plans', 10)

        # Service clients
        self.validation_client = self.create_client(
            CommandValidation, '/validate_command')

        # Task execution state
        self.current_task = None
        self.task_queue = []

        self.get_logger().info('Task Planning Node initialized')

    def command_callback(self, msg):
        """Callback for parsed commands"""
        # Validate the command first
        if self.validate_command(msg.original_command):
            # Plan the task
            task_plan = self.create_task_plan(msg)
            if task_plan:
                self.plan_pub.publish(task_plan)
                self.execute_task_plan(task_plan)

    def validate_command(self, command):
        """Validate command using service"""
        while not self.validation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Validation service not available, waiting...')

        request = CommandValidation.Request()
        request.command = command

        future = self.validation_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response.is_valid:
            self.get_logger().info(f'Command validated: {response.reason}')
            return True
        else:
            self.get_logger().warn(f'Command validation failed: {response.reason}')
            return False

    def create_task_plan(self, parsed_cmd):
        """Create a task plan from parsed command"""
        task_plan = TaskPlan()
        task_plan.header.stamp = self.get_clock().now().to_msg()
        task_plan.header.frame_id = 'map'
        task_plan.original_command = parsed_cmd.original_command

        # Based on action type, create appropriate task steps
        if parsed_cmd.action_type == 'navigation':
            # Navigation task
            nav_step = TaskStep()
            nav_step.action_type = 'navigation'
            nav_step.target_location = parsed_cmd.target_location
            nav_step.parameters = parsed_cmd.parameters
            task_plan.steps.append(nav_step)

        elif parsed_cmd.action_type == 'manipulation':
            # Manipulation task - might need navigation first
            if parsed_cmd.target_location:
                # Navigate to location first
                nav_step = TaskStep()
                nav_step.action_type = 'navigation'
                nav_step.target_location = parsed_cmd.target_location
                nav_step.parameters = parsed_cmd.parameters
                task_plan.steps.append(nav_step)

            # Then manipulate object
            manip_step = TaskStep()
            manip_step.action_type = 'manipulation'
            manip_step.target_object = parsed_cmd.target_object
            manip_step.parameters = parsed_cmd.parameters
            task_plan.steps.append(manip_step)

        elif parsed_cmd.action_type == 'combined':
            # Combined task - navigate and manipulate
            nav_step = TaskStep()
            nav_step.action_type = 'navigation'
            nav_step.target_location = parsed_cmd.target_location
            task_plan.steps.append(nav_step)

            manip_step = TaskStep()
            manip_step.action_type = 'manipulation'
            manip_step.target_object = parsed_cmd.target_object
            task_plan.steps.append(manip_step)

        else:
            self.get_logger().warn(f'Unknown action type: {parsed_cmd.action_type}')
            return None

        return task_plan

    def execute_task_plan(self, task_plan):
        """Execute the task plan"""
        self.get_logger().info(f'Executing task plan with {len(task_plan.steps)} steps')

        for i, step in enumerate(task_plan.steps):
            self.get_logger().info(f'Executing step {i+1}/{len(task_plan.steps)}: {step.action_type}')

            # In a real system, you would:
            # 1. Send the step to appropriate action server
            # 2. Wait for completion
            # 3. Handle errors
            # 4. Update system state

            # For this example, we'll just log the step
            self.execute_task_step(step)

        self.get_logger().info('Task plan execution completed')

    def execute_task_step(self, step):
        """Execute a single task step"""
        # This would interface with the appropriate subsystem
        if step.action_type == 'navigation':
            self.get_logger().info(f'Navigating to {step.target_location}')
        elif step.action_type == 'manipulation':
            self.get_logger().info(f'Manipulating {step.target_object}')
        else:
            self.get_logger().warn(f'Unknown step type: {step.action_type}')

def main(args=None):
    rclpy.init(args=args)
    node = TaskPlanningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down task planning node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation Integration

### 10. Create Gazebo Launch File

Create a launch file to start the complete simulation:

```xml
<!-- File: humanoid_simulation/launch/complete_simulation.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    gazebo_ros_package_dir = get_package_share_directory('gazebo_ros')
    humanoid_description_dir = get_package_share_directory('humanoid_description')

    return LaunchDescription([
        # Start Gazebo server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_package_dir, 'launch', 'gzserver.launch.py')
            )
        ),

        # Start Gazebo client
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_package_dir, 'launch', 'gzclient.launch.py')
            )
        ),

        # Spawn the humanoid robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'humanoid_robot',
                '-topic', 'robot_description',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.0'
            ],
            output='screen'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'robot_description':
                    open(os.path.join(humanoid_description_dir, 'urdf', 'humanoid.urdf')).read()}
            ]
        ),

        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
        )
    ])
```

### 11. Create Complete System Launch File

Create a launch file to start the entire autonomous humanoid system:

```xml
<!-- File: humanoid_bringup/launch/autonomous_humanoid.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    humanoid_voice_dir = get_package_share_directory('humanoid_voice_control')
    humanoid_simulation_dir = get_package_share_directory('humanoid_simulation')

    return LaunchDescription([
        # Start the simulation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(humanoid_simulation_dir, 'launch', 'complete_simulation.launch.py')
            )
        ),

        # Start voice control system
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(humanoid_voice_dir, 'launch', 'voice_system.launch.py')
            )
        ),

        # Start perception system
        Node(
            package='humanoid_perception',
            executable='perception_node',
            name='perception_node',
            output='screen'
        ),

        # Start navigation system
        Node(
            package='humanoid_navigation',
            executable='navigation_node',
            name='navigation_node',
            output='screen'
        ),

        # Start manipulation system
        Node(
            package='humanoid_manipulation',
            executable='manipulation_node',
            name='manipulation_node',
            output='screen'
        ),

        # Start task planning system
        Node(
            package='humanoid_planning',
            executable='task_planning_node',
            name='task_planning_node',
            output='screen'
        )
    ])
```

## Testing the Complete System

### 12. Run the Complete System

Now let's test the complete system by launching everything:

```bash
# Source the workspace
cd ~/autonomous_humanoid_ws
source install/setup.bash

# Launch the complete autonomous humanoid system
ros2 launch humanoid_bringup autonomous_humanoid.launch.py
```

### 13. Test Voice Commands

Once the system is running, you can test voice commands. In a separate terminal:

```bash
# Test navigation commands
ros2 topic pub /voice_commands std_msgs/String "data: 'Go to the kitchen'"

# Test manipulation commands
ros2 topic pub /voice_commands std_msgs/String "data: 'Pick up the red cup'"

# Test combined commands
ros2 topic pub /voice_commands std_msgs/String "data: 'Go to the living room and find the blue ball'"
```

### 14. Monitor System Status

Monitor the system to see how different components interact:

```bash
# Monitor voice commands
ros2 topic echo /voice_commands

# Monitor parsed commands
ros2 topic echo /parsed_commands

# Monitor task plans
ros2 topic echo /task_plans

# Monitor detected objects
ros2 topic echo /detected_objects

# Check system services
ros2 service list

# Check system actions
ros2 action list
```

## Troubleshooting Common Issues

### 15. Common Issues and Solutions

1. **Audio not being captured**:
   - Check microphone permissions
   - Verify audio device is selected as default
   - Test with `arecord -d 5 test.wav`

2. **LLM integration errors**:
   - Verify OpenAI API key is set in environment
   - Check network connectivity
   - Confirm billing is active for your OpenAI account

3. **Navigation failures**:
   - Check if Nav2 is properly installed
   - Verify map and localization are working
   - Ensure robot has proper transforms (tf)

4. **Object detection not working**:
   - Check camera is publishing images
   - Verify camera calibration
   - Ensure perception node is running

## Key Concepts

- **System Integration**: All components must work together seamlessly
- **Real-time Processing**: System must respond to commands within acceptable timeframes
- **Component Coordination**: Different subsystems need to communicate effectively
- **Error Handling**: Robust error handling across all components
- **Safety Considerations**: Validation of commands before execution
- **Modular Design**: Each component should be independently testable

## Practical Exercises

1. **Extend the voice command vocabulary**: Add support for new commands like "turn left", "turn right", "stop", etc.

2. **Improve object detection**: Integrate a real object detection model (YOLO, SSD, etc.) instead of the simple color-based detection.

3. **Add more locations**: Expand the location map with more destinations in the simulated environment.

4. **Implement error recovery**: Add mechanisms to handle and recover from failed navigation or manipulation attempts.

5. **Add safety checks**: Implement more sophisticated safety validation before executing commands.

6. **Create a GUI interface**: Build a simple GUI that displays system status and allows manual command input.

## Common Failure Modes

- **Integration Failures**: Components not properly communicating with each other
- **Timing Issues**: Delays causing poor user experience or system instability
- **Resource Exhaustion**: High computational requirements affecting performance
- **Safety Violations**: Actions executed without proper safety checks
- **Error Propagation**: Failures in one component affecting others
- **Communication Failures**: ROS topics/services not connecting properly
- **Perception Failures**: Object detection or scene understanding errors
- **Action Execution Failures**: Navigation or manipulation errors