# Troubleshooting: ROS 2 Common Issues

## Common Installation and Setup Issues

### ROS 2 Installation Problems

**Issue**: Cannot source ROS 2 setup files
- **Symptoms**: `source /opt/ros/humble/setup.bash` returns "No such file or directory"
- **Solutions**:
  1. Verify ROS 2 Humble is installed: `ls /opt/ros/`
  2. Check correct ROS 2 distribution is installed (Humble Hawksbill for this book)
  3. If using a different distribution, adjust the path accordingly (e.g., `/opt/ros/iron/`)

**Issue**: Python modules not found
- **Symptoms**: `ImportError: No module named 'rclpy'`
- **Solutions**:
  1. Ensure ROS 2 environment is sourced: `source /opt/ros/humble/setup.bash`
  2. Check if Python packages are in the correct path: `echo $PYTHONPATH`
  3. Install Python dependencies: `sudo apt install python3-colcon-common-extensions`

### Network and Communication Issues

**Issue**: Nodes cannot communicate across machines
- **Symptoms**: Publishers and subscribers on different machines cannot see each other
- **Solutions**:
  1. Ensure both machines are on the same network
  2. Check firewall settings (DDS uses ports 7400-7499 by default)
  3. Set ROS domain ID: `export ROS_DOMAIN_ID=42` (same on all machines)
  4. Verify network configuration: `echo $ROS_LOCALHOST_ONLY` (should be 0 or unset)

**Issue**: QoS profile mismatch
- **Symptoms**: Publishers and subscribers not communicating despite correct topic names
- **Solutions**:
  1. Ensure QoS profiles match between publishers and subscribers
  2. Use compatible QoS settings (RELIABLE with RELIABLE, not BEST_EFFORT)
  3. Check durability settings (TRANSIENT_LOCAL vs VOLATILE)

## Node and Communication Issues

### Node Lifecycle Problems

**Issue**: Node fails to initialize
- **Symptoms**: `rclpy.init()` fails or node constructor throws an exception
- **Solutions**:
  1. Verify ROS 2 environment is sourced
  2. Check if another ROS 2 process is running that might conflict
  3. Ensure proper initialization: `rclpy.init(args=args)` before creating nodes

**Issue**: Node doesn't respond to Ctrl+C
- **Symptoms**: Node continues running after pressing Ctrl+C
- **Solutions**:
  1. Ensure proper cleanup in main function:
     ```python
     try:
         rclpy.spin(node)
     except KeyboardInterrupt:
         pass
     finally:
         node.destroy_node()
         rclpy.shutdown()
     ```

### Publisher and Subscriber Issues

**Issue**: Publisher not sending messages
- **Symptoms**: No messages appear when using `ros2 topic echo`
- **Solutions**:
  1. Check if timer is properly created and running
  2. Verify publisher is created with correct topic name
  3. Ensure `rclpy.spin()` is called to process callbacks

**Issue**: Subscriber not receiving messages
- **Symptoms**: Callback function never executes
- **Solutions**:
  1. Verify topic names match exactly (case-sensitive)
  2. Check message types are compatible
  3. Ensure QoS profiles are compatible between publisher and subscriber
  4. Verify node is spinning: `rclpy.spin(node)`

## Service and Action Issues

### Service Communication Problems

**Issue**: Service client times out waiting for service
- **Symptoms**: `while not self.cli.wait_for_service(timeout_sec=1.0):` loops indefinitely
- **Solutions**:
  1. Verify service name matches exactly
  2. Ensure service server is running before client
  3. Check if service is advertised with correct type

**Issue**: Service request fails
- **Symptoms**: Service call returns without response or throws exception
- **Solutions**:
  1. Check service callback function returns the expected response type
  2. Verify request/response message types match
  3. Ensure service callback doesn't throw exceptions

### Action Communication Problems

**Issue**: Action client cannot connect to action server
- **Symptoms**: `self._action_client.wait_for_server()` times out
- **Solutions**:
  1. Verify action name matches exactly
  2. Ensure action server is running before client
  3. Check action type matches between client and server

**Issue**: Action goal execution fails
- **Symptoms**: Goal is rejected or fails without clear error
- **Solutions**:
  1. Check action server properly handles goal requests
  2. Verify goal, feedback, and result message types
  3. Ensure action server calls appropriate callbacks (succeed, cancel, etc.)

## URDF and Robot Description Issues

### URDF Parsing Errors

**Issue**: Robot description fails to load
- **Symptoms**: Error messages about URDF parsing, robot not visible in RViz
- **Solutions**:
  1. Validate XML syntax using an XML validator
  2. Check all joint and link names are unique
  3. Verify all parent-child relationships are valid
  4. Ensure all required elements are present (mass, inertia for links)

**Issue**: Robot appears incorrectly in RViz
- **Symptoms**: Parts of robot are missing or positioned incorrectly
- **Solutions**:
  1. Check origin transformations in joints
  2. Verify joint types and limits are correct
  3. Ensure visual and collision elements are properly defined

### Robot State Publisher Issues

**Issue**: Joint states not updating in visualization
- **Symptoms**: Robot appears rigid or joints don't move as expected
- **Solutions**:
  1. Verify joint_state_publisher is running
  2. Check that joint states are published to `/joint_states` topic
  3. Ensure robot description parameter is correctly set in robot_state_publisher

## Performance and Resource Issues

### Memory and CPU Usage

**Issue**: High CPU usage
- **Symptoms**: System becomes slow, ROS 2 processes consume high CPU
- **Solutions**:
  1. Check timer frequencies - avoid unnecessarily high rates
  2. Reduce message publishing frequency if possible
  3. Use appropriate QoS settings (BEST_EFFORT for non-critical data)

**Issue**: Memory leaks
- **Symptoms**: Memory usage increases over time
- **Solutions**:
  1. Properly destroy nodes: `node.destroy_node()`
  2. Clean up subscriptions, publishers, and services when done
  3. Use context managers or try/finally blocks for cleanup

### Communication Performance

**Issue**: Message delays or dropped messages
- **Symptoms**: High latency in communication, missing messages
- **Solutions**:
  1. Check network performance between machines
  2. Adjust QoS settings (history depth, reliability)
  3. Monitor system resources for bottlenecks

## Debugging Strategies

### Using ROS 2 Command Line Tools

**Check node connectivity**:
```bash
# List all nodes
ros2 node list

# Check node info
ros2 node info <node_name>

# List topics
ros2 topic list

# Echo a topic
ros2 topic echo <topic_name> <message_type>
```

**Check service and action status**:
```bash
# List services
ros2 service list

# Call a service
ros2 service call <service_name> <service_type> <request_data>

# List actions
ros2 action list
```

### Logging and Debugging

**Enable detailed logging**:
```bash
# Set log level
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG

# Or set for specific nodes
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=INFO
```

**Use launch file debugging**:
```python
# Add logging to launch files
from launch.actions import LogInfo

# Example of debugging in launch file
LogInfo(msg=['Launching node: ', launch_configuration_variable])
```

## Key Concepts

- **ROS_DOMAIN_ID**: Environment variable to isolate ROS 2 networks
- **QoS Compatibility**: Publishers/subscribers need compatible QoS profiles
- **Node Lifecycle**: Proper initialization and cleanup is essential
- **Message Types**: Publishers and subscribers must use compatible message types
- **URDF Validation**: XML syntax and semantic correctness required
- **Resource Management**: Proper cleanup prevents memory leaks

## Practical Exercises

1. Create a diagnostic node that checks for common ROS 2 issues
2. Implement a health monitoring system for your ROS 2 nodes
3. Develop a debugging launch file with detailed logging
4. Create a script that validates URDF files before loading

## Common Failure Modes

- **Environment Not Sourced**: Forgetting to source ROS 2 setup files
- **QoS Mismatch**: Incompatible QoS profiles preventing communication
- **Resource Leaks**: Not properly destroying nodes or handles
- **Network Configuration**: Firewall or domain ID issues in multi-machine setups
- **Message Type Errors**: Incompatible message types between nodes
- **Initialization Order**: Starting clients before servers or actions
- **URDF Syntax Errors**: Malformed XML preventing robot description loading
- **Timer Issues**: Improper timer configuration causing performance problems

## Key Concepts

- **ROS_DOMAIN_ID**: Environment variable to isolate ROS 2 networks
- **QoS Compatibility**: Publishers/subscribers need compatible QoS profiles
- **Node Lifecycle**: Proper initialization and cleanup is essential
- **Message Types**: Publishers and subscribers must use compatible message types
- **URDF Validation**: XML syntax and semantic correctness required
- **Resource Management**: Proper cleanup prevents memory leaks

## Practical Exercises

1. Create a diagnostic node that checks for common ROS 2 issues
2. Implement a health monitoring system for your ROS 2 nodes
3. Develop a debugging launch file with detailed logging
4. Create a script that validates URDF files before loading

## Common Failure Modes (Continued)

- **Environment Not Sourced**: Forgetting to source ROS 2 setup files
- **QoS Mismatch**: Incompatible QoS profiles preventing communication
- **Resource Leaks**: Not properly destroying nodes or handles
- **Network Configuration**: Firewall or domain ID issues in multi-machine setups
- **Message Type Errors**: Incompatible message types between nodes
- **Initialization Order**: Starting clients before servers or actions
- **URDF Syntax Errors**: Malformed XML preventing robot description loading
- **Timer Issues**: Improper timer configuration causing performance problems