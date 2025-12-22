---
sidebar_position: 4
---

# Capstone Code Examples: Complete Integration Code

## Overview

This module provides complete, working code examples for the autonomous humanoid system. These examples demonstrate how to integrate all components from the previous modules into a functioning voice-controlled robot system. Each example includes detailed explanations and can be used as a foundation for your own implementations.

## Complete Voice Command Integration Example

### Voice Command Processing Node

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
import os

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

        # Initialize OpenAI with API key from environment
        openai.api_key = os.getenv('OPENAI_API_KEY', 'your-openai-key')

        self.get_logger().info('Voice Command Node initialized')

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

                    # Log the recognized command
                    self.get_logger().info(f'Recognized command: {text}')

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
            # Define the system prompt for command parsing
            system_prompt = """You are a command parser for a humanoid robot.
            Parse the user's natural language command into a structured format.
            Return JSON with action_type, target_object, target_location, and parameters.
            Action types: 'navigation', 'manipulation', 'combined', 'query', 'other'.
            Example: {'action_type': 'navigation', 'target_location': 'kitchen',
            'target_object': null, 'parameters': {}}"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=150
            )

            # Parse the response
            result = json.loads(response.choices[0].message.content)

            # Create parsed command message
            parsed_cmd = ParsedCommand()
            parsed_cmd.action_type = result.get('action_type', 'other')
            parsed_cmd.target_object = result.get('target_object', '')
            parsed_cmd.target_location = result.get('target_location', '')
            parsed_cmd.parameters = json.dumps(result.get('parameters', {}))
            parsed_cmd.original_command = text
            parsed_cmd.confidence = 0.9  # Assuming high confidence for LLM output

            self.get_logger().info(f'Parsed command: {parsed_cmd.action_type} to {parsed_cmd.target_location} for {parsed_cmd.target_object}')
            return parsed_cmd

        except json.JSONDecodeError:
            self.get_logger().error(f'LLM response not valid JSON: {response.choices[0].message.content}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error parsing command with LLM: {e}')
            return None

    def validate_command_callback(self, request, response):
        """Validate if a command is safe to execute"""
        # Define dangerous keywords that should not be executed
        dangerous_keywords = [
            'harm', 'damage', 'break', 'destroy', 'hurt', 'injure',
            'attack', 'kill', 'unsafe', 'dangerous', 'emergency'
        ]

        # Check if command contains dangerous keywords
        is_safe = not any(keyword in request.command.lower() for keyword in dangerous_keywords)

        if is_safe:
            response.is_valid = True
            response.reason = "Command is safe"
        else:
            response.is_valid = False
            response.reason = "Command contains potentially unsafe keywords"

        self.get_logger().info(f'Command validation: {request.command} -> {response.is_valid}')
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

## Complete Perception System Example

### Advanced Perception Node with Isaac ROS Integration

```python
#!/usr/bin/env python3
# File: humanoid_perception/humanoid_perception/advanced_perception_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point, Pose, TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import cv2
import numpy as np
from humanoid_msgs.msg import DetectedObject, SemanticScene
from builtin_interfaces.msg import Time
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Vector3, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

class AdvancedPerceptionNode(Node):
    def __init__(self):
        super().__init__('advanced_perception_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # TF broadcaster for transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)

        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detected_objects', 10)
        self.semantic_pub = self.create_publisher(
            SemanticScene, '/semantic_scene', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/object_markers', 10)

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = 640
        self.image_height = 480

        # Object detection parameters
        self.confidence_threshold = 0.5
        self.min_object_area = 500

        # Detected objects storage
        self.detected_objects = []
        self.object_colors = {}  # Track colors for object persistence

        # Initialize YOLO model (simplified - in practice use Isaac ROS DNN)
        # For this example, we'll use OpenCV's DNN module
        try:
            # Load YOLO model (you would download these files in a real implementation)
            # self.net = cv2.dnn.readNetFromDarknet('yolo.cfg', 'yolo.weights')
            # For this example, we'll use a simpler approach
            self.get_logger().info('Perception node initialized with basic detection')
        except Exception as e:
            self.get_logger().warn(f'Could not load advanced detection model: {e}')
            self.get_logger().info('Using basic color-based detection instead')

        self.get_logger().info('Advanced Perception Node initialized')

    def camera_info_callback(self, msg):
        """Callback for camera info"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.image_width = msg.width
        self.image_height = msg.height

    def pointcloud_callback(self, msg):
        """Callback for point cloud data"""
        # Process point cloud data for 3D object information
        # This would typically interface with Isaac ROS point cloud processing
        pass

    def image_callback(self, msg):
        """Callback for image data"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection
            detections = self.detect_objects(cv_image)

            # Create and publish detections
            detection_msg = self.create_detection_message(detections, msg.header)
            self.detection_pub.publish(detection_msg)

            # Create and publish semantic scene
            semantic_msg = self.create_semantic_scene(detections, msg.header)
            self.semantic_pub.publish(semantic_msg)

            # Publish visualization markers
            marker_array = self.create_visualization_markers(detections, msg.header)
            self.marker_pub.publish(marker_array)

            # Publish TF transforms for detected objects
            self.publish_object_transforms(detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, image):
        """Perform object detection on image"""
        detections = []

        # For this example, we'll implement a more sophisticated approach
        # that combines color-based detection with basic shape analysis

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'red2': ([170, 100, 100], [180, 255, 255]),  # Red wraps around HSV
            'blue': ([100, 100, 100], [130, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'yellow': ([20, 100, 100], [40, 255, 255])
        }

        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)

            # Create mask for this color
            if color_name == 'red2':
                mask = cv2.inRange(hsv, lower, upper)
                mask = mask + cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            else:
                mask = cv2.inRange(hsv, lower, upper)

            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_object_area:  # Filter small areas
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Determine object type based on shape and size
                    aspect_ratio = float(w) / h
                    extent = float(area) / (w * h)

                    if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                        obj_type = f'{color_name}_object'
                    elif aspect_ratio < 0.7:
                        obj_type = f'{color_name}_tall_object'
                    else:
                        obj_type = f'{color_name}_wide_object'

                    detection = {
                        'class': obj_type,
                        'confidence': min(0.9, area / 10000.0),  # Normalize confidence based on size
                        'bbox': [x, y, w, h],
                        'center': [center_x, center_y],
                        'area': area
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
                vision_detection.bbox.center.x = float(detection['center'][0])
                vision_detection.bbox.center.y = float(detection['center'][1])
                vision_detection.bbox.size_x = float(detection['bbox'][2])
                vision_detection.bbox.size_y = float(detection['bbox'][3])

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

                # Convert 2D image coordinates to 3D world coordinates if camera matrix is available
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    # Simple depth assumption for example - in practice you'd use depth data
                    # For this example, we'll assume all objects are at 1 meter depth
                    image_point = np.array([[detection['center'][0], detection['center'][1]]], dtype=np.float32)
                    world_point = cv2.undistortPoints(image_point, self.camera_matrix, self.dist_coeffs)

                    # Convert to 3D assuming Z=1 (in front of camera)
                    obj.position.x = world_point[0][0][0] * 1.0  # Scale by assumed depth
                    obj.position.y = world_point[0][0][1] * 1.0
                    obj.position.z = 1.0  # Assumed depth
                else:
                    # Fallback to 2D coordinates
                    obj.position.x = detection['center'][0]
                    obj.position.y = detection['center'][1]
                    obj.position.z = 0.0

                obj.dimensions.x = detection['bbox'][2]  # Width
                obj.dimensions.y = detection['bbox'][3]  # Height
                obj.dimensions.z = 0.1  # Assumed depth

                semantic_scene.objects.append(obj)

        return semantic_scene

    def create_visualization_markers(self, detections, header):
        """Create visualization markers for detected objects"""
        marker_array = MarkerArray()

        for i, detection in enumerate(detections):
            if detection['confidence'] > self.confidence_threshold:
                # Create text marker for object label
                text_marker = Marker()
                text_marker.header = header
                text_marker.ns = "object_labels"
                text_marker.id = i * 2
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD

                # Position text above the object
                if self.camera_matrix is not None:
                    # Convert image coordinates to approximate world coordinates
                    text_marker.pose.position.x = detection['center'][0] * 0.001  # Scale factor
                    text_marker.pose.position.y = detection['center'][1] * 0.001
                    text_marker.pose.position.z = 1.2  # Slightly above the object
                else:
                    text_marker.pose.position.x = detection['center'][0]
                    text_marker.pose.position.y = detection['center'][1]
                    text_marker.pose.position.z = 0.1

                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.1  # Text size
                text_marker.color.a = 1.0
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.text = f"{detection['class']}: {detection['confidence']:.2f}"

                marker_array.markers.append(text_marker)

                # Create box marker for bounding box
                box_marker = Marker()
                box_marker.header = header
                box_marker.ns = "object_boxes"
                box_marker.id = i * 2 + 1
                box_marker.type = Marker.LINE_STRIP
                box_marker.action = Marker.ADD

                # Define the 4 corners of the bounding box
                x, y, w, h = detection['bbox']
                corners = [
                    Point(x=float(x), y=float(y), z=0.01),
                    Point(x=float(x + w), y=float(y), z=0.01),
                    Point(x=float(x + w), y=float(y + h), z=0.01),
                    Point(x=float(x), y=float(y + h), z=0.01),
                    Point(x=float(x), y=float(y), z=0.01)  # Close the loop
                ]

                box_marker.points = corners
                box_marker.pose.orientation.w = 1.0
                box_marker.scale.x = 0.02  # Line width
                box_marker.color.a = 0.8
                box_marker.color.r = 1.0
                box_marker.color.g = 1.0
                box_marker.color.b = 0.0

                marker_array.markers.append(box_marker)

        return marker_array

    def publish_object_transforms(self, detections, header):
        """Publish TF transforms for detected objects"""
        for i, detection in enumerate(detections):
            if detection['confidence'] > self.confidence_threshold:
                t = TransformStamped()

                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = header.frame_id
                t.child_frame_id = f"detected_{detection['class']}_{i}"

                # Set transform position
                if self.camera_matrix is not None:
                    # Convert image coordinates to approximate world coordinates
                    t.transform.translation.x = detection['center'][0] * 0.001
                    t.transform.translation.y = detection['center'][1] * 0.001
                    t.transform.translation.z = 1.0  # Assumed depth
                else:
                    t.transform.translation.x = detection['center'][0] * 0.001
                    t.transform.translation.y = detection['center'][1] * 0.001
                    t.transform.translation.z = 0.1

                # No rotation for detected objects
                t.transform.rotation.w = 1.0

                # Publish the transform
                self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down advanced perception node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Complete Navigation System Example

### Navigation Integration with Path Planning

```python
#!/usr/bin/env python3
# File: humanoid_navigation/humanoid_navigation/advanced_navigation_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from humanoid_msgs.msg import ParsedCommand, NavigationGoal, SemanticScene
from humanoid_msgs.srv import NavigateToPose
from nav2_msgs.action import NavigateToPose as NavigateToPoseAction
from rclpy.action import ActionClient
from tf2_ros import TransformListener, Buffer
from builtin_interfaces.msg import Duration
import json
import math

class AdvancedNavigationNode(Node):
    def __init__(self):
        super().__init__('advanced_navigation_node')

        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPoseAction, 'navigate_to_pose')

        # TF listener for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            ParsedCommand, '/parsed_commands', self.command_callback, 10)
        self.nav_goal_pub = self.create_publisher(
            NavigationGoal, '/navigation_goals', 10)
        self.semantic_scene_sub = self.create_subscription(
            SemanticScene, '/semantic_scene', self.semantic_scene_callback, 10)

        # Service for navigation
        self.nav_srv = self.create_service(
            NavigateToPose, '/navigate_to_pose', self.navigate_to_pose_callback)

        # Known locations mapping with poses
        self.location_map = {
            'kitchen': Pose(position=Point(x=2.0, y=1.0, z=0.0),
                           orientation=Quaternion(w=1.0)),
            'living_room': Pose(position=Point(x=-1.0, y=0.0, z=0.0),
                              orientation=Quaternion(w=1.0)),
            'bedroom': Pose(position=Point(x=0.0, y=-2.0, z=0.0),
                           orientation=Quaternion(w=1.0)),
            'office': Pose(position=Point(x=1.5, y=-1.0, z=0.0),
                         orientation=Quaternion(w=1.0)),
            'entrance': Pose(position=Point(x=0.0, y=2.0, z=0.0),
                           orientation=Quaternion(w=1.0)),
            'charging_station': Pose(position=Point(x=-2.0, y=0.0, z=0.0),
                                   orientation=Quaternion(w=1.0))
        }

        # Semantic scene storage
        self.semantic_scene = None
        self.last_navigation_time = self.get_clock().now()

        self.get_logger().info('Advanced Navigation Node initialized')

    def semantic_scene_callback(self, msg):
        """Callback for semantic scene updates"""
        self.semantic_scene = msg
        self.get_logger().info(f'Updated semantic scene with {len(msg.objects)} objects')

    def command_callback(self, msg):
        """Callback for parsed commands"""
        if msg.action_type in ['navigation', 'combined']:
            self.handle_navigation_command(msg)

    def handle_navigation_command(self, cmd_msg):
        """Handle navigation commands"""
        target_location = cmd_msg.target_location.lower()

        # Try to find the location in our map
        if target_location in self.location_map:
            target_pose = self.location_map[target_location]
            self.get_logger().info(f'Navigating to known location: {target_location}')
        else:
            # Try to find the location as an object in the scene
            target_pose = self.find_object_in_scene(target_location)
            if target_pose is None:
                self.get_logger().warn(f'Unknown location/object: {target_location}')
                return

        # Create navigation goal
        goal_msg = NavigateToPoseAction.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose = target_pose

        # Send navigation goal
        self.send_navigation_goal(goal_msg)

        # Publish navigation goal message
        nav_goal_msg = NavigationGoal()
        nav_goal_msg.target_location = target_location
        nav_goal_msg.target_pose = goal_msg.pose
        nav_goal_msg.original_command = cmd_msg.original_command
        self.nav_goal_pub.publish(nav_goal_msg)

    def find_object_in_scene(self, object_name):
        """Find an object in the semantic scene and return its pose"""
        if self.semantic_scene is None:
            return None

        for obj in self.semantic_scene.objects:
            if object_name.lower() in obj.name.lower():
                # Create a pose for the object
                pose = Pose()
                pose.position = obj.position
                pose.orientation.w = 1.0  # Default orientation
                return pose

        return None

    def send_navigation_goal(self, goal_msg):
        """Send navigation goal to Nav2"""
        self.nav_client.wait_for_server()

        # Send the goal asynchronously
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected by server')
            return

        self.get_logger().info('Navigation goal accepted by server')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle navigation result"""
        try:
            result = future.result().result
            self.get_logger().info(f'Navigation completed with result: {result}')
        except Exception as e:
            self.get_logger().error(f'Navigation failed with error: {e}')

    def navigate_to_pose_callback(self, request, response):
        """Service callback for navigation"""
        try:
            goal_msg = NavigateToPoseAction.Goal()
            goal_msg.pose = request.target_pose

            # Check if we can navigate (not too frequent)
            current_time = self.get_clock().now()
            time_diff = (current_time - self.last_navigation_time).nanoseconds / 1e9
            if time_diff < 1.0:  # Minimum 1 second between navigation requests
                response.success = False
                response.message = "Navigation request too frequent"
                return response

            self.nav_client.wait_for_server()
            send_goal_future = self.nav_client.send_goal_async(goal_msg)

            # For this example, we'll return success immediately
            # In a real system, you'd wait for completion or implement async handling
            response.success = True
            response.message = "Navigation goal sent successfully"
            self.last_navigation_time = current_time

        except Exception as e:
            response.success = False
            response.message = f'Error during navigation: {str(e)}'

        return response

    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        dz = pose1.position.z - pose2.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down advanced navigation node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Complete Manipulation System Example

### Manipulation Node with Grasp Planning

```python
#!/usr/bin/env python3
# File: humanoid_manipulation/humanoid_manipulation/advanced_manipulation_node.py

import rclpy
from rclpy.node import Node
from humanoid_msgs.msg import ParsedCommand, DetectedObject, SemanticScene
from humanoid_msgs.srv import ManipulateObject
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from builtin_interfaces.msg import Duration
import json
import math

class AdvancedManipulationNode(Node):
    def __init__(self):
        super().__init__('advanced_manipulation_node')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            ParsedCommand, '/parsed_commands', self.command_callback, 10)
        self.objects_sub = self.create_subscription(
            SemanticScene, '/semantic_scene', self.semantic_scene_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointTrajectoryControllerState, '/joint_states', self.joint_state_callback, 10)

        # Service for manipulation
        self.manip_srv = self.create_service(
            ManipulateObject, '/manipulate_object', self.manipulate_callback)

        # Trajectory publisher for arm control
        self.arm_traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # Detected objects storage
        self.semantic_scene = None
        self.joint_states = None

        # Robot arm parameters (simplified)
        self.arm_joints = ['shoulder_pan_joint', 'shoulder_lift_joint',
                          'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.gripper_joints = ['left_gripper_joint', 'right_gripper_joint']

        # Default arm configuration
        self.default_arm_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.default_gripper_pose = [0.0, 0.0]  # Open

        self.get_logger().info('Advanced Manipulation Node initialized')

    def semantic_scene_callback(self, msg):
        """Callback for semantic scene updates"""
        self.semantic_scene = msg
        self.get_logger().info(f'Updated semantic scene with {len(msg.objects)} objects')

    def joint_state_callback(self, msg):
        """Callback for joint states"""
        self.joint_states = msg

    def command_callback(self, msg):
        """Callback for parsed commands"""
        if msg.action_type in ['manipulation', 'combined']:
            self.handle_manipulation_command(msg)

    def handle_manipulation_command(self, cmd_msg):
        """Handle manipulation commands"""
        target_object = cmd_msg.target_object.lower()

        if not self.semantic_scene:
            self.get_logger().warn('No semantic scene available for manipulation')
            return

        # Find the object in detected objects
        object_to_manipulate = None
        for obj in self.semantic_scene.objects:
            if target_object in obj.name.lower():
                object_to_manipulate = obj
                break

        if object_to_manipulate:
            self.get_logger().info(f'Attempting to manipulate {object_to_manipulate.name}')

            # Plan and execute manipulation
            success = self.execute_manipulation(object_to_manipulate, cmd_msg.original_command)

            if success:
                self.get_logger().info(f'Successfully manipulated {object_to_manipulate.name}')
            else:
                self.get_logger().warn(f'Failed to manipulate {object_to_manipulate.name}')
        else:
            self.get_logger().warn(f'Object {target_object} not found in scene')

    def execute_manipulation(self, obj, command):
        """Execute manipulation based on command and object"""
        # Parse the command to determine the action
        command_lower = command.lower()

        if 'pick' in command_lower or 'grasp' in command_lower or 'grab' in command_lower:
            return self.execute_pick(obj)
        elif 'place' in command_lower or 'put' in command_lower:
            # For place, we need a target location
            target_location = self.extract_location_from_command(command)
            if target_location:
                return self.execute_place(obj, target_location)
            else:
                return self.execute_place(obj, 'default')
        elif 'move' in command_lower:
            target_location = self.extract_location_from_command(command)
            if target_location:
                return self.execute_move(obj, target_location)

        self.get_logger().warn(f'Unknown manipulation action in command: {command}')
        return False

    def extract_location_from_command(self, command):
        """Extract target location from command"""
        # Simple keyword-based extraction
        command_lower = command.lower()

        location_keywords = {
            'kitchen': ['kitchen', 'counter', 'table'],
            'living_room': ['living room', 'couch', 'sofa'],
            'bedroom': ['bedroom', 'bed'],
            'office': ['office', 'desk'],
            'table': ['table', 'surface'],
            'shelf': ['shelf', 'cabinet']
        }

        for location, keywords in location_keywords.items():
            for keyword in keywords:
                if keyword in command_lower:
                    return location

        return None

    def execute_pick(self, obj):
        """Execute pick action for an object"""
        try:
            # 1. Plan approach trajectory to the object
            approach_pose = self.calculate_approach_pose(obj)
            if not approach_pose:
                return False

            # 2. Move arm to approach position
            if not self.move_arm_to_pose(approach_pose):
                return False

            # 3. Calculate grasp pose (slightly above the object)
            grasp_pose = self.calculate_grasp_pose(obj)
            if not grasp_pose:
                return False

            # 4. Move arm to grasp position
            if not self.move_arm_to_pose(grasp_pose):
                return False

            # 5. Close gripper to grasp the object
            if not self.close_gripper():
                return False

            # 6. Lift the object slightly
            lift_pose = self.calculate_lift_pose(grasp_pose)
            if not self.move_arm_to_pose(lift_pose):
                return False

            self.get_logger().info(f'Successfully picked up {obj.name}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error during pick operation: {e}')
            return False

    def execute_place(self, obj, location):
        """Execute place action for an object"""
        try:
            # Determine placement location based on the location parameter
            placement_pose = self.calculate_placement_pose(location)
            if not placement_pose:
                return False

            # Move to placement location
            if not self.move_arm_to_pose(placement_pose):
                return False

            # Open gripper to release object
            if not self.open_gripper():
                return False

            # Move arm up slightly to clear the placed object
            clear_pose = self.calculate_clear_pose(placement_pose)
            if not self.move_arm_to_pose(clear_pose):
                return False

            self.get_logger().info(f'Successfully placed {obj.name} at {location}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error during place operation: {e}')
            return False

    def execute_move(self, obj, target_location):
        """Execute move action for an object"""
        try:
            # First pick up the object
            if not self.execute_pick(obj):
                return False

            # Then place it at the target location
            return self.execute_place(obj, target_location)

        except Exception as e:
            self.get_logger().error(f'Error during move operation: {e}')
            return False

    def calculate_approach_pose(self, obj):
        """Calculate approach pose for an object"""
        # Approach from a safe distance above and in front of the object
        approach_pose = Pose()
        approach_pose.position.x = obj.position.x
        approach_pose.position.y = obj.position.y
        approach_pose.position.z = obj.position.z + 0.2  # 20cm above
        approach_pose.orientation.w = 1.0  # Default orientation

        return approach_pose

    def calculate_grasp_pose(self, obj):
        """Calculate grasp pose for an object"""
        # Position gripper at the object's location
        grasp_pose = Pose()
        grasp_pose.position.x = obj.position.x
        grasp_pose.position.y = obj.position.y
        grasp_pose.position.z = obj.position.z + obj.dimensions.z/2  # Half the height
        grasp_pose.orientation.w = 1.0  # Default orientation

        return grasp_pose

    def calculate_lift_pose(self, grasp_pose):
        """Calculate lift pose after grasping"""
        lift_pose = Pose()
        lift_pose.position.x = grasp_pose.position.x
        lift_pose.position.y = grasp_pose.position.y
        lift_pose.position.z = grasp_pose.position.z + 0.1  # Lift 10cm
        lift_pose.orientation.w = grasp_pose.orientation.w

        return lift_pose

    def calculate_placement_pose(self, location):
        """Calculate placement pose based on location"""
        # Simplified placement positions for different locations
        placement_positions = {
            'kitchen': Point(x=1.5, y=0.5, z=0.8),  # Kitchen counter
            'living_room': Point(x=-0.5, y=-0.5, z=0.6),  # Coffee table
            'bedroom': Point(x=0.5, y=-1.5, z=0.7),  # Bedside table
            'office': Point(x=1.0, y=-0.5, z=0.75),  # Desk
            'table': Point(x=0.0, y=0.0, z=0.75),  # Default table
            'shelf': Point(x=1.0, y=0.0, z=1.2),  # Shelf
            'default': Point(x=0.0, y=0.0, z=0.75)  # Default position
        }

        if location in placement_positions:
            placement_pose = Pose()
            placement_pose.position = placement_positions[location]
            placement_pose.orientation.w = 1.0
            return placement_pose
        else:
            self.get_logger().warn(f'Unknown placement location: {location}, using default')
            default_pose = Pose()
            default_pose.position = placement_positions['default']
            default_pose.orientation.w = 1.0
            return default_pose

    def calculate_clear_pose(self, placement_pose):
        """Calculate clear pose after placing"""
        clear_pose = Pose()
        clear_pose.position.x = placement_pose.position.x
        clear_pose.position.y = placement_pose.position.y
        clear_pose.position.z = placement_pose.position.z + 0.1  # Lift slightly
        clear_pose.orientation.w = placement_pose.orientation.w

        return clear_pose

    def move_arm_to_pose(self, pose):
        """Move the robot arm to a specific pose"""
        try:
            # This is a simplified trajectory planning
            # In a real system, you'd use inverse kinematics to calculate joint angles
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.arm_joints

            # Create a single trajectory point (simplified)
            point = JointTrajectoryPoint()

            # For this example, we'll use a simple approach
            # In reality, you'd calculate the joint angles using IK
            point.positions = self.default_arm_pose.copy()  # Placeholder

            # Set timing
            point.time_from_start.sec = 2
            point.time_from_start.nanosec = 0

            trajectory_msg.points.append(point)

            # Publish the trajectory
            self.arm_traj_pub.publish(trajectory_msg)

            # Wait for completion (simplified)
            self.get_clock().sleep_for(Duration(nanosec=2000000000))  # 2 seconds

            self.get_logger().info(f'Moved arm to pose: ({pose.position.x}, {pose.position.y}, {pose.position.z})')
            return True

        except Exception as e:
            self.get_logger().error(f'Error moving arm to pose: {e}')
            return False

    def close_gripper(self):
        """Close the robot gripper"""
        try:
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.gripper_joints

            point = JointTrajectoryPoint()
            # Close gripper (simplified - actual values depend on gripper design)
            point.positions = [0.02, 0.02]  # Closed position
            point.time_from_start.sec = 1
            point.time_from_start.nanosec = 0

            trajectory_msg.points.append(point)

            self.gripper_pub.publish(trajectory_msg)

            # Wait for completion
            self.get_clock().sleep_for(Duration(nanosec=1000000000))  # 1 second

            self.get_logger().info('Closed gripper')
            return True

        except Exception as e:
            self.get_logger().error(f'Error closing gripper: {e}')
            return False

    def open_gripper(self):
        """Open the robot gripper"""
        try:
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.gripper_joints

            point = JointTrajectoryPoint()
            # Open gripper
            point.positions = [0.08, 0.08]  # Open position
            point.time_from_start.sec = 1
            point.time_from_start.nanosec = 0

            trajectory_msg.points.append(point)

            self.gripper_pub.publish(trajectory_msg)

            # Wait for completion
            self.get_clock().sleep_for(Duration(nanosec=1000000000))  # 1 second

            self.get_logger().info('Opened gripper')
            return True

        except Exception as e:
            self.get_logger().error(f'Error opening gripper: {e}')
            return False

    def manipulate_callback(self, request, response):
        """Service callback for manipulation"""
        try:
            # In a real system, you would implement the full manipulation pipeline
            if request.action == 'pick':
                # Find the object in the semantic scene
                if self.semantic_scene:
                    for obj in self.semantic_scene.objects:
                        if request.object_name.lower() in obj.name.lower():
                            response.success = self.execute_pick(obj)
                            break
                    else:
                        response.success = False
                        response.message = f'Object {request.object_name} not found'
                else:
                    response.success = False
                    response.message = 'No semantic scene available'

            elif request.action == 'place':
                # For place, we'd need to find a carried object or implement differently
                response.success = True  # Simplified for example
                response.message = f'Successfully placed object at {request.location}'

            elif request.action == 'move':
                response.success = True  # Simplified for example
                response.message = f'Successfully moved {request.object_name} to {request.location}'

            else:
                response.success = False
                response.message = f'Unknown action: {request.action}'

        except Exception as e:
            response.success = False
            response.message = f'Error during manipulation: {str(e)}'

        if response.success:
            response.message = f'Successfully executed {request.action} action'
        else:
            response.message = f'Failed to execute {request.action} action: {response.message}'

        self.get_logger().info(response.message)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedManipulationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down advanced manipulation node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Complete System Integration Example

### Main Integration Node

```python
#!/usr/bin/env python3
# File: humanoid_system_integration/humanoid_system_integration/integration_node.py

import rclpy
from rclpy.node import Node
from humanoid_msgs.msg import ParsedCommand, TaskPlan, TaskStep
from humanoid_msgs.srv import CommandValidation
from std_msgs.msg import String, Bool
from builtin_interfaces.msg import Time
import json
import time

class SystemIntegrationNode(Node):
    def __init__(self):
        super().__init__('system_integration_node')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            ParsedCommand, '/parsed_commands', self.command_callback, 10)
        self.plan_sub = self.create_subscription(
            TaskPlan, '/task_plans', self.plan_callback, 10)

        self.system_status_pub = self.create_publisher(
            String, '/system_status', 10)
        self.task_status_pub = self.create_publisher(
            String, '/task_status', 10)

        # Service clients for various subsystems
        self.nav_client = self.create_client(
            CommandValidation, '/validate_command')  # Placeholder for navigation service
        self.manip_client = self.create_client(
            CommandValidation, '/validate_command')  # Placeholder for manipulation service

        # System state
        self.system_ready = False
        self.current_task = None
        self.task_queue = []
        self.subsystem_status = {
            'voice': False,
            'perception': False,
            'navigation': False,
            'manipulation': False,
            'planning': False
        }

        # Timer for system monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_system)

        self.get_logger().info('System Integration Node initialized')

    def command_callback(self, msg):
        """Callback for parsed commands"""
        self.get_logger().info(f'Received command: {msg.original_command} ({msg.action_type})')

        # Validate command
        if self.validate_command(msg):
            # Create and execute task plan
            task_plan = self.create_task_plan(msg)
            if task_plan:
                self.execute_task_plan(task_plan)

    def plan_callback(self, msg):
        """Callback for task plans"""
        self.get_logger().info(f'Received task plan with {len(msg.steps)} steps')

    def validate_command(self, parsed_cmd):
        """Validate command using subsystem validation"""
        # For this example, we'll do basic validation
        if not parsed_cmd.original_command.strip():
            self.get_logger().warn('Empty command received')
            return False

        # Check if system is ready
        if not self.system_ready:
            self.get_logger().warn('System not ready to process commands')
            return False

        # Check if all required subsystems are available
        required_subsystems = []
        if parsed_cmd.action_type in ['navigation', 'combined']:
            required_subsystems.append('navigation')
        if parsed_cmd.action_type in ['manipulation', 'combined']:
            required_subsystems.append('manipulation')

        for subsystem in required_subsystems:
            if not self.subsystem_status.get(subsystem, False):
                self.get_logger().warn(f'Required subsystem {subsystem} not available')
                return False

        self.get_logger().info(f'Command validated: {parsed_cmd.original_command}')
        return True

    def create_task_plan(self, parsed_cmd):
        """Create a task plan from parsed command"""
        task_plan = TaskPlan()
        task_plan.header.stamp = self.get_clock().now().to_msg()
        task_plan.header.frame_id = 'map'
        task_plan.original_command = parsed_cmd.original_command
        task_plan.command_type = parsed_cmd.action_type

        # Based on action type, create appropriate task steps
        if parsed_cmd.action_type == 'navigation':
            # Navigation task
            nav_step = TaskStep()
            nav_step.action_type = 'navigation'
            nav_step.target_location = parsed_cmd.target_location
            nav_step.parameters = parsed_cmd.parameters
            nav_step.description = f'Navigate to {parsed_cmd.target_location}'
            task_plan.steps.append(nav_step)

        elif parsed_cmd.action_type == 'manipulation':
            # Manipulation task - might need navigation first
            if parsed_cmd.target_location:
                # Navigate to location first
                nav_step = TaskStep()
                nav_step.action_type = 'navigation'
                nav_step.target_location = parsed_cmd.target_location
                nav_step.parameters = parsed_cmd.parameters
                nav_step.description = f'Navigate to {parsed_cmd.target_location}'
                task_plan.steps.append(nav_step)

            # Then manipulate object
            manip_step = TaskStep()
            manip_step.action_type = 'manipulation'
            manip_step.target_object = parsed_cmd.target_object
            manip_step.parameters = parsed_cmd.parameters
            manip_step.description = f'Manipulate {parsed_cmd.target_object}'
            task_plan.steps.append(manip_step)

        elif parsed_cmd.action_type == 'combined':
            # Combined task - navigate and manipulate
            nav_step = TaskStep()
            nav_step.action_type = 'navigation'
            nav_step.target_location = parsed_cmd.target_location
            nav_step.description = f'Navigate to {parsed_cmd.target_location}'
            task_plan.steps.append(nav_step)

            manip_step = TaskStep()
            manip_step.action_type = 'manipulation'
            manip_step.target_object = parsed_cmd.target_object
            manip_step.description = f'Manipulate {parsed_cmd.target_object}'
            task_plan.steps.append(manip_step)

        elif parsed_cmd.action_type == 'query':
            # Query task - perception only
            query_step = TaskStep()
            query_step.action_type = 'perception'
            query_step.target_object = parsed_cmd.target_object
            query_step.description = f'Query for {parsed_cmd.target_object}'
            task_plan.steps.append(query_step)

        else:
            self.get_logger().warn(f'Unknown action type: {parsed_cmd.action_type}')
            return None

        # Publish the task plan
        plan_pub = self.create_publisher(TaskPlan, '/task_plans', 10)
        plan_pub.publish(task_plan)

        self.get_logger().info(f'Created task plan with {len(task_plan.steps)} steps')
        return task_plan

    def execute_task_plan(self, task_plan):
        """Execute the task plan step by step"""
        self.get_logger().info(f'Executing task plan: {task_plan.original_command}')
        self.current_task = task_plan

        # Update task status
        status_msg = String()
        status_msg.data = f'Executing task: {task_plan.original_command}'
        self.task_status_pub.publish(status_msg)

        # Execute each step in the plan
        for i, step in enumerate(task_plan.steps):
            self.get_logger().info(f'Executing step {i+1}/{len(task_plan.steps)}: {step.description}')

            success = self.execute_task_step(step)
            if not success:
                self.get_logger().error(f'Task step {i+1} failed: {step.description}')
                # For this example, we'll continue with other steps, but in a real system
                # you might want to implement different failure handling strategies
                continue

        # Task completed
        self.get_logger().info('Task plan execution completed')
        self.current_task = None

        # Update status
        status_msg = String()
        status_msg.data = 'Task completed'
        self.task_status_pub.publish(status_msg)

    def execute_task_step(self, step):
        """Execute a single task step"""
        try:
            if step.action_type == 'navigation':
                return self.execute_navigation_step(step)
            elif step.action_type == 'manipulation':
                return self.execute_manipulation_step(step)
            elif step.action_type == 'perception':
                return self.execute_perception_step(step)
            else:
                self.get_logger().warn(f'Unknown step type: {step.action_type}')
                return False

        except Exception as e:
            self.get_logger().error(f'Error executing task step: {e}')
            return False

    def execute_navigation_step(self, step):
        """Execute navigation step"""
        self.get_logger().info(f'Navigating to {step.target_location}')

        # In a real system, you would call the navigation service
        # For this example, we'll simulate the navigation
        time.sleep(2)  # Simulate navigation time

        self.get_logger().info(f'Reached {step.target_location}')
        return True

    def execute_manipulation_step(self, step):
        """Execute manipulation step"""
        self.get_logger().info(f'Manipulating {step.target_object}')

        # In a real system, you would call the manipulation service
        # For this example, we'll simulate the manipulation
        time.sleep(3)  # Simulate manipulation time

        self.get_logger().info(f'Finished manipulating {step.target_object}')
        return True

    def execute_perception_step(self, step):
        """Execute perception step"""
        self.get_logger().info(f'Perceiving {step.target_object}')

        # In a real system, you would call the perception service
        # For this example, we'll simulate the perception
        time.sleep(1)  # Simulate perception time

        self.get_logger().info(f'Finished perceiving {step.target_object}')
        return True

    def monitor_system(self):
        """Monitor system status and subsystem availability"""
        # In a real system, you would check the actual status of each subsystem
        # For this example, we'll simulate status checking

        # Simulate checking if subsystems are responding
        self.subsystem_status['voice'] = True  # Assume voice system is always available
        self.subsystem_status['perception'] = True  # Assume perception is available
        self.subsystem_status['navigation'] = True  # Assume navigation is available
        self.subsystem_status['manipulation'] = True  # Assume manipulation is available
        self.subsystem_status['planning'] = True  # Assume planning is available

        # Update system readiness based on subsystem status
        self.system_ready = all(self.subsystem_status.values())

        # Publish system status
        status_msg = String()
        status_msg.data = f"System Ready: {self.system_ready}, Subsystems: {self.subsystem_status}"
        self.system_status_pub.publish(status_msg)

        # Log system status periodically
        if self.system_ready:
            self.get_logger().info('System is ready to process commands')
        else:
            self.get_logger().info(f'System not ready. Status: {self.subsystem_status}')

def main(args=None):
    rclpy.init(args=args)
    node = SystemIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down system integration node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files and Configuration

### Main Launch File for Complete System

```xml
<!-- File: humanoid_bringup/launch/complete_autonomous_humanoid.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_namespace = LaunchConfiguration('robot_namespace', default='humanoid_robot')

    # Get package directories
    gazebo_ros_package_dir = get_package_share_directory('gazebo_ros')
    humanoid_description_dir = get_package_share_directory('humanoid_description')
    humanoid_voice_dir = get_package_share_directory('humanoid_voice_control')
    humanoid_simulation_dir = get_package_share_directory('humanoid_simulation')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='humanoid_robot',
            description='Robot namespace'
        ),

        # Start Gazebo server with a world file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_package_dir, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={
                'world': PathJoinSubstitution([
                    humanoid_simulation_dir,
                    'worlds',
                    'humanoid_world.sdf'
                ])
            }.items()
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
                '-entity', [robot_namespace],
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
                {'use_sim_time': use_sim_time},
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
                {'use_sim_time': use_sim_time}
            ]
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
            executable='advanced_perception_node',
            name='advanced_perception_node',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),

        # Start navigation system
        Node(
            package='humanoid_navigation',
            executable='advanced_navigation_node',
            name='advanced_navigation_node',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),

        # Start manipulation system
        Node(
            package='humanoid_manipulation',
            executable='advanced_manipulation_node',
            name='advanced_manipulation_node',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),

        # Start task planning system
        Node(
            package='humanoid_planning',
            executable='task_planning_node',
            name='task_planning_node',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),

        # Start system integration node
        Node(
            package='humanoid_system_integration',
            executable='integration_node',
            name='integration_node',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        )
    ])
```

## Testing and Validation Scripts

### System Test Script

```python
#!/usr/bin/env python3
# File: test_system_integration.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from humanoid_msgs.msg import ParsedCommand
import time

class SystemTestNode(Node):
    def __init__(self):
        super().__init__('system_test_node')

        # Publishers for testing
        self.voice_cmd_pub = self.create_publisher(String, '/voice_commands', 10)
        self.parsed_cmd_pub = self.create_publisher(ParsedCommand, '/parsed_commands', 10)

        self.get_logger().info('System Test Node initialized')

        # Schedule test commands
        self.timer = self.create_timer(5.0, self.run_tests)
        self.test_counter = 0
        self.test_commands = [
            'Go to the kitchen',
            'Find the red cup',
            'Pick up the blue ball',
            'Go to the living room',
            'Put the ball on the table'
        ]

    def run_tests(self):
        """Run system integration tests"""
        if self.test_counter < len(self.test_commands):
            command = self.test_commands[self.test_counter]

            # Publish test command
            cmd_msg = String()
            cmd_msg.data = command
            self.voice_cmd_pub.publish(cmd_msg)

            self.get_logger().info(f'Sent test command: {command}')

            self.test_counter += 1
        else:
            self.get_logger().info('All tests completed')
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = SystemTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down system test node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Concepts

- **System Integration**: All components must work together seamlessly
- **Real-time Processing**: System must respond to commands within acceptable timeframes
- **Component Coordination**: Different subsystems need to communicate effectively
- **Error Handling**: Robust error handling across all components
- **Safety Considerations**: Validation of commands before execution
- **Modular Design**: Each component should be independently testable
- **ROS 2 Communication**: Proper use of topics, services, and actions
- **State Management**: Tracking system state and task progress

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