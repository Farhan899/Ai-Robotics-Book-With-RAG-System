# Code Examples: LLM Integration and ROS 2 Action Mapping

## Large Language Model Integration

### OpenAI GPT Integration for Command Understanding

```python
import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import ParsedCommand
import json
import os


class LLMCommandParserNode(Node):

    def __init__(self):
        super().__init__('llm_command_parser_node')

        # Publishers and Subscribers
        self.command_sub = self.create_subscription(
            String, 'voice_commands', self.command_callback, 10)
        self.parsed_pub = self.create_publisher(
            ParsedCommand, 'parsed_commands', 10)

        # Initialize OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            self.get_logger().error('OPENAI_API_KEY environment variable not set')
            return

        # Define system prompt for command parsing
        self.system_prompt = """
        You are a command parser for a robotic system. Your job is to interpret natural language commands
        and convert them into structured robot commands.

        Available actions:
        - navigate_to: Move the robot to a location
        - grasp_object: Pick up an object
        - place_object: Place an object at a location
        - detect_object: Find an object in the environment
        - follow_person: Follow a person
        - report_status: Report robot status

        Response format:
        {
            "action": "action_name",
            "target_object": "object_name_or_null",
            "target_location": "location_name_or_null",
            "parameters": {"key": "value"}
        }

        Examples:
        Input: "Please go to the kitchen and get me a cup"
        Output: {"action": "navigate_to", "target_location": "kitchen", "target_object": null, "parameters": {}}

        Input: "Pick up the red cup"
        Output: {"action": "grasp_object", "target_object": "red cup", "target_location": null, "parameters": {}}
        """

        self.get_logger().info('LLM Command Parser Node initialized')

    def command_callback(self, msg):
        """Process incoming voice commands using LLM"""
        command_text = msg.data
        self.get_logger().info(f'Processing command with LLM: {command_text}')

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": command_text}
                ],
                temperature=0.1,  # Low temperature for more consistent parsing
                max_tokens=150
            )

            # Extract and parse the response
            response_text = response.choices[0].message.content.strip()

            # Clean up the response (remove any markdown formatting)
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ```

            # Parse the JSON response
            parsed_data = json.loads(response_text)

            # Create and publish parsed command
            parsed_cmd = ParsedCommand()
            parsed_cmd.action = parsed_data.get('action', 'unknown')
            parsed_cmd.target_object = parsed_data.get('target_object', '')
            parsed_cmd.target_location = parsed_data.get('target_location', '')
            parsed_cmd.raw_command = command_text
            parsed_cmd.parameters = json.dumps(parsed_data.get('parameters', {}))

            self.parsed_pub.publish(parsed_cmd)
            self.get_logger().info(f'LLM parsed command: {parsed_cmd.action}')

        except json.JSONDecodeError:
            self.get_logger().error(f'LLM returned invalid JSON: {response_text}')
        except Exception as e:
            self.get_logger().error(f'Error processing command with LLM: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LLMCommandParserNode()

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

## Hugging Face Transformers Integration

### Local LLM for Command Understanding

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import ParsedCommand
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import json


class LocalLLMCommandParserNode(Node):

    def __init__(self):
        super().__init__('local_llm_command_parser_node')

        # Publishers and Subscribers
        self.command_sub = self.create_subscription(
            String, 'voice_commands', self.command_callback, 10)
        self.parsed_pub = self.create_publisher(
            ParsedCommand, 'parsed_commands', 10)

        # Load a pre-trained model (using a smaller model for efficiency)
        model_name = "microsoft/DialoGPT-medium"  # Example model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create text generation pipeline
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        # Define command templates
        self.command_template = """
        You are a robot command parser. Convert the following command into structured format:
        Command: {command}

        Available actions: navigate_to, grasp_object, place_object, detect_object, follow_person, report_status

        Format your response as JSON:
        {{
            "action": "action_name",
            "target_object": "object_name_or_null",
            "target_location": "location_name_or_null",
            "parameters": {{"key": "value"}}
        }}
        """

        self.get_logger().info('Local LLM Command Parser Node initialized')

    def command_callback(self, msg):
        """Process incoming voice commands using local LLM"""
        command_text = msg.data
        self.get_logger().info(f'Processing command with local LLM: {command_text}')

        try:
            # Create prompt
            prompt = self.command_template.format(command=command_text)

            # Generate response
            response = self.generator(
                prompt,
                max_length=len(self.tokenizer.encode(prompt)) + 100,
                num_return_sequences=1,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Extract the generated text
            generated_text = response[0]['generated_text'][len(prompt):]

            # Try to find JSON in the response
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_str = generated_text[json_start:json_end]
                parsed_data = json.loads(json_str)

                # Create and publish parsed command
                parsed_cmd = ParsedCommand()
                parsed_cmd.action = parsed_data.get('action', 'unknown')
                parsed_cmd.target_object = parsed_data.get('target_object', '')
                parsed_cmd.target_location = parsed_data.get('target_location', '')
                parsed_cmd.raw_command = command_text
                parsed_cmd.parameters = json.dumps(parsed_data.get('parameters', {}))

                self.parsed_pub.publish(parsed_cmd)
                self.get_logger().info(f'Local LLM parsed command: {parsed_cmd.action}')
            else:
                self.get_logger().warn(f'Could not extract JSON from LLM response: {generated_text}')

        except json.JSONDecodeError:
            self.get_logger().error(f'LLM returned invalid JSON: {generated_text}')
        except Exception as e:
            self.get_logger().error(f'Error processing command with local LLM: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LocalLLMCommandParserNode()

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

## Vision-Language Integration

### Vision-Grounded Language Understanding

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vla_interfaces.msg import ParsedCommand
from cv_bridge import CvBridge
import cv2
import numpy as np
import openai
import os
import json


class VisionGroundedLLMNode(Node):

    def __init__(self):
        super().__init__('vision_grounded_llm_node')

        # Publishers and Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(
            String, 'voice_commands', self.command_callback, 10)
        self.parsed_pub = self.create_publisher(
            ParsedCommand, 'parsed_commands', 10)

        # Initialize components
        self.bridge = CvBridge()
        self.current_image = None
        self.image_timestamp = None

        # Initialize OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')

        self.get_logger().info('Vision-Grounded LLM Node initialized')

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process voice command with vision grounding"""
        command_text = msg.data
        self.get_logger().info(f'Processing vision-grounded command: {command_text}')

        if self.current_image is None:
            self.get_logger().warn('No current image for vision grounding')
            # Fall back to text-only processing
            self.process_text_only_command(command_text)
            return

        try:
            # Convert image to base64 for API
            _, buffer = cv2.imencode('.jpg', self.current_image)
            image_base64 = buffer.tobytes().hex()

            # Use OpenAI's vision-capable model (GPT-4 Vision)
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Based on this image, interpret the command: '{command_text}'. What should the robot do? If the command refers to specific objects, identify their locations and characteristics. Respond in JSON format with action, target_object, and target_location."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)

                # Create and publish parsed command
                parsed_cmd = ParsedCommand()
                parsed_cmd.action = parsed_data.get('action', 'unknown')
                parsed_cmd.target_object = parsed_data.get('target_object', '')
                parsed_cmd.target_location = parsed_data.get('target_location', '')
                parsed_cmd.raw_command = command_text
                parsed_cmd.parameters = json.dumps(parsed_data.get('parameters', {}))

                self.parsed_pub.publish(parsed_cmd)
                self.get_logger().info(f'Vision-grounded command: {parsed_cmd.action}')

        except Exception as e:
            self.get_logger().error(f'Error in vision-grounded processing: {e}')
            # Fall back to text-only processing
            self.process_text_only_command(command_text)

    def process_text_only_command(self, command_text):
        """Process command without vision (fallback)"""
        # Simple fallback processing
        parsed_cmd = ParsedCommand()
        parsed_cmd.action = 'unknown'
        parsed_cmd.target_object = ''
        parsed_cmd.target_location = ''
        parsed_cmd.raw_command = command_text
        parsed_cmd.parameters = '{}'

        # Simple keyword-based parsing as fallback
        command_lower = command_text.lower()

        if any(word in command_lower for word in ['go to', 'navigate to', 'move to']):
            parsed_cmd.action = 'navigate_to'
        elif any(word in command_lower for word in ['pick up', 'grasp', 'get', 'take']):
            parsed_cmd.action = 'grasp_object'
        elif any(word in command_lower for word in ['place', 'put', 'set down']):
            parsed_cmd.action = 'place_object'
        elif any(word in command_lower for word in ['find', 'look for', 'locate']):
            parsed_cmd.action = 'detect_object'

        self.parsed_pub.publish(parsed_cmd)
        self.get_logger().info(f'Text-only fallback: {parsed_cmd.action}')


def main(args=None):
    rclpy.init(args=args)
    node = VisionGroundedLLMNode()

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

## ROS 2 Action Mapping and Execution

### Action Planner and Executor

```python
import rclpy
from rclpy.node import Node
from vla_interfaces.msg import ParsedCommand, TaskPlan, ExecutableAction
from geometry_msgs.msg import Pose, Point
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import json
import threading


class VLAActionExecutorNode(Node):

    def __init__(self):
        super().__init__('vla_action_executor_node')

        # Publishers and Subscribers
        self.plan_sub = self.create_subscription(
            TaskPlan, 'task_plans', self.plan_callback, 10)
        self.status_pub = self.create_publisher(
            String, 'action_status', 10)

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Task execution state
        self.current_plan = None
        self.current_task_index = 0
        self.is_executing = False
        self.execution_lock = threading.Lock()

        # Timer for executing tasks
        self.timer = self.create_timer(0.1, self.execute_next_task)

        self.get_logger().info('VLA Action Executor Node initialized')

    def plan_callback(self, msg):
        """Receive and queue task plans for execution"""
        with self.execution_lock:
            if not self.is_executing:
                self.current_plan = msg
                self.current_task_index = 0
                self.is_executing = True
                self.get_logger().info(f'New plan received with {len(msg.tasks)} tasks')
            else:
                self.get_logger().info('Plan queued, current plan still executing')

    def execute_next_task(self):
        """Execute the next task in the current plan"""
        if not self.is_executing or self.current_plan is None:
            return

        if self.current_task_index >= len(self.current_plan.tasks):
            # Plan completed
            self.plan_completed()
            return

        current_task = self.current_plan.tasks[self.current_task_index]
        self.get_logger().info(f'Executing task {self.current_task_index + 1}: {current_task.action}')

        # Execute based on task type
        if current_task.action == 'navigate_to':
            self.execute_navigation_task(current_task)
        elif current_task.action == 'grasp_object':
            self.execute_grasp_task(current_task)
        elif current_task.action == 'place_object':
            self.execute_place_task(current_task)
        elif current_task.action == 'detect_object':
            self.execute_detection_task(current_task)
        else:
            self.get_logger().warn(f'Unknown task action: {current_task.action}')
            self.task_completed()

    def execute_navigation_task(self, task):
        """Execute navigation task"""
        try:
            # Parse target location from parameters
            params = json.loads(task.parameters) if task.parameters else {}

            if 'location' in params:
                location = params['location']

                # Create navigation goal
                goal = NavigateToPose.Goal()
                goal.pose.header.frame_id = 'map'
                goal.pose.pose.position.x = float(location[0])
                goal.pose.pose.position.y = float(location[1])
                goal.pose.pose.position.z = float(location[2])
                goal.pose.pose.orientation.w = 1.0  # Default orientation

                # Send navigation goal
                self.nav_client.wait_for_server()
                future = self.nav_client.send_goal_async(goal)
                future.add_done_callback(self.navigation_result_callback)

                # Publish status
                status_msg = String()
                status_msg.data = f'navigating_to: [{location[0]}, {location[1]}, {location[2]}]'
                self.status_pub.publish(status_msg)

            else:
                self.get_logger().warn('Navigation task missing location parameter')
                self.task_completed()

        except Exception as e:
            self.get_logger().error(f'Error executing navigation task: {e}')
            self.task_completed()

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info('Navigation goal accepted')
                # Wait for result
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(self.navigation_finished_callback)
            else:
                self.get_logger().warn('Navigation goal rejected')
                self.task_completed()
        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            self.task_completed()

    def navigation_finished_callback(self, future):
        """Handle navigation completion"""
        try:
            result = future.result().result
            self.get_logger().info(f'Navigation completed with result: {result}')
        except Exception as e:
            self.get_logger().error(f'Navigation result error: {e}')
        finally:
            self.task_completed()

    def execute_grasp_task(self, task):
        """Execute grasp task (placeholder implementation)"""
        try:
            params = json.loads(task.parameters) if task.parameters else {}

            if 'object' in params:
                object_name = params['object']
                self.get_logger().info(f'Attempting to grasp object: {object_name}')

                # Publish status
                status_msg = String()
                status_msg.data = f'grasping_object: {object_name}'
                self.status_pub.publish(status_msg)

                # Simulate grasp operation completion
                # In real implementation, you would use robot manipulation interfaces
                self.get_logger().info('Grasp operation completed')

            else:
                self.get_logger().warn('Grasp task missing object parameter')

        except Exception as e:
            self.get_logger().error(f'Error executing grasp task: {e}')
        finally:
            self.task_completed()

    def execute_place_task(self, task):
        """Execute place task (placeholder implementation)"""
        try:
            params = json.loads(task.parameters) if task.parameters else {}

            if 'object' in params:
                object_name = params['object']
                self.get_logger().info(f'Attempting to place object: {object_name}')

                # Publish status
                status_msg = String()
                status_msg.data = f'placing_object: {object_name}'
                self.status_pub.publish(status_msg)

                # Simulate place operation completion
                self.get_logger().info('Place operation completed')

            else:
                self.get_logger().warn('Place task missing object parameter')

        except Exception as e:
            self.get_logger().error(f'Error executing place task: {e}')
        finally:
            self.task_completed()

    def execute_detection_task(self, task):
        """Execute detection task (placeholder implementation)"""
        try:
            params = json.loads(task.parameters) if task.parameters else {}

            if 'object' in params:
                object_name = params['object']
                self.get_logger().info(f'Attempting to detect object: {object_name}')

                # Publish status
                status_msg = String()
                status_msg.data = f'detecting_object: {object_name}'
                self.status_pub.publish(status_msg)

                # Simulate detection operation completion
                self.get_logger().info('Detection operation completed')

            else:
                self.get_logger().warn('Detection task missing object parameter')

        except Exception as e:
            self.get_logger().error(f'Error executing detection task: {e}')
        finally:
            self.task_completed()

    def task_completed(self):
        """Mark current task as completed and move to next"""
        with self.execution_lock:
            self.current_task_index += 1
            if self.current_task_index >= len(self.current_plan.tasks):
                self.plan_completed()
            else:
                self.get_logger().info(f'Task completed. Moving to task {self.current_task_index + 1}')

    def plan_completed(self):
        """Handle plan completion"""
        with self.execution_lock:
            if self.current_plan:
                self.get_logger().info(f'Plan completed with {len(self.current_plan.tasks)} tasks')

                # Publish completion status
                status_msg = String()
                status_msg.data = 'plan_completed'
                self.status_pub.publish(status_msg)

                self.current_plan = None
                self.current_task_index = 0
                self.is_executing = False

    def cancel_current_plan(self):
        """Cancel the current plan execution"""
        with self.execution_lock:
            self.current_plan = None
            self.current_task_index = 0
            self.is_executing = False
            self.get_logger().info('Current plan cancelled')


def main(args=None):
    rclpy.init(args=args)
    node = VLAActionExecutorNode()

    # Use multi-threaded executor to handle callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        executor.shutdown()


if __name__ == '__main__':
    main()
```

## Whisper Integration for Speech-to-Text

### OpenAI Whisper Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
from sensor_msgs.msg import AudioData as SensorAudioData
import torch
import whisper
import numpy as np
import io
import wave
import threading
import queue


class WhisperSTTNode(Node):

    def __init__(self):
        super().__init__('whisper_stt_node')

        # Publishers and Subscribers
        self.audio_sub = self.create_subscription(
            SensorAudioData, 'audio', self.audio_callback, 10)
        self.command_pub = self.create_publisher(
            String, 'voice_commands', 10)

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")  # Use "tiny", "base", "small", "medium", or "large"
        self.get_logger().info('Whisper model loaded')

        # Audio processing queue
        self.audio_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_audio_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('Whisper STT Node initialized')

    def audio_callback(self, msg):
        """Receive audio data and add to processing queue"""
        try:
            # Convert audio data to format expected by Whisper
            audio_data = np.frombuffer(msg.data, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Add to processing queue
            self.audio_queue.put(audio_float)

        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def process_audio_queue(self):
        """Process audio from queue using Whisper"""
        while rclpy.ok():
            try:
                # Get audio from queue (with timeout to allow thread to exit)
                audio_float = self.audio_queue.get(timeout=1.0)

                # Process with Whisper
                result = self.model.transcribe(audio_float)
                text = result['text'].strip()

                if text:  # Only publish non-empty results
                    self.get_logger().info(f'Recognized: {text}')

                    # Publish recognized command
                    cmd_msg = String()
                    cmd_msg.data = text
                    self.command_pub.publish(cmd_msg)

            except queue.Empty:
                # Timeout - continue loop
                continue
            except Exception as e:
                self.get_logger().error(f'Error in audio processing: {e}')

    def destroy_node(self):
        """Clean up before node destruction"""
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)  # Wait up to 2 seconds
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WhisperSTTNode()

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

## Complete VLA System Integration

### Main VLA System Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_interfaces.msg import ParsedCommand, TaskPlan
from geometry_msgs.msg import Twist
import threading
import time


class VLAMasterNode(Node):

    def __init__(self):
        super().__init__('vla_master_node')

        # Publishers
        self.status_pub = self.create_publisher(String, 'vla_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'voice_commands', self.voice_command_callback, 10)
        self.parsed_sub = self.create_subscription(
            ParsedCommand, 'parsed_commands', self.parsed_command_callback, 10)
        self.plan_sub = self.create_subscription(
            TaskPlan, 'task_plans', self.task_plan_callback, 10)

        # System state
        self.system_state = 'idle'  # idle, processing, executing
        self.last_command = ''
        self.active_plan = None

        # Status reporting timer
        self.status_timer = self.create_timer(5.0, self.report_status)

        self.get_logger().info('VLA Master Node initialized')

    def voice_command_callback(self, msg):
        """Handle incoming voice commands"""
        command = msg.data
        self.get_logger().info(f'Received voice command: {command}')

        self.last_command = command
        self.system_state = 'processing'

        # Update status
        status_msg = String()
        status_msg.data = f'processing_command: {command}'
        self.status_pub.publish(status_msg)

    def parsed_command_callback(self, msg):
        """Handle parsed commands"""
        self.get_logger().info(f'Parsed command: {msg.action} for object "{msg.target_object}" at location "{msg.target_location}"')

        # Could trigger immediate action or planning based on complexity
        if msg.action == 'report_status':
            self.report_robot_status()

    def task_plan_callback(self, msg):
        """Handle incoming task plans"""
        self.get_logger().info(f'Received task plan with {len(msg.tasks)} tasks')
        self.active_plan = msg
        self.system_state = 'executing'

        # Update status
        status_msg = String()
        status_msg.data = f'executing_plan_with_{len(msg.tasks)}_tasks'
        self.status_pub.publish(status_msg)

    def report_status(self):
        """Periodically report system status"""
        status_msg = String()
        status_msg.data = f'system_state: {self.system_state}, last_command: {self.last_command[:50] if self.last_command else "none"}'
        self.status_pub.publish(status_msg)

    def report_robot_status(self):
        """Report current robot status"""
        # In a real implementation, this would gather actual robot status
        status_msg = String()
        status_msg.data = 'robot_status: battery_85%, position_known, systems_operational'
        self.status_pub.publish(status_msg)

    def emergency_stop(self):
        """Emergency stop function"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().warn('Emergency stop activated')


def main(args=None):
    rclpy.init(args=args)
    node = VLAMasterNode()

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

## Key Concepts

- **Large Language Models (LLMs)**: AI models for natural language understanding
- **Vision-Language Integration**: Combining visual and textual information
- **Speech-to-Text (STT)**: Converting speech to text using systems like Whisper
- **Action Mapping**: Connecting language concepts to robot actions
- **Task Planning**: Decomposing high-level commands into executable steps
- **ROS Actions**: Asynchronous goal-oriented communication with feedback
- **Vision Grounding**: Connecting language to visual elements in the environment
- **Real-time Processing**: Handling inputs with low latency requirements

## Practical Exercises

1. Implement a complete VLA system with your preferred LLM
2. Integrate visual grounding with language understanding
3. Create a robust action mapping system for your robot platform
4. Implement error handling and recovery in the VLA pipeline
5. Add support for multi-modal inputs (speech, text, gestures)

## Common Failure Modes

- **LLM Hallucination**: Language models generating incorrect or fabricated information
- **Audio Quality Issues**: Poor audio affecting speech recognition accuracy
- **Context Confusion**: LLMs losing track of conversation context
- **Timing Issues**: Delays in processing affecting natural interaction
- **Ambiguity Resolution**: Difficulty in resolving ambiguous commands
- **Resource Exhaustion**: High computational requirements affecting performance
- **Safety Violations**: Actions executed without proper safety checks
- **Misinterpretation**: Natural language not correctly mapped to robot actions