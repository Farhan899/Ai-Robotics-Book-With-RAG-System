# Troubleshooting: Digital Twin and Simulation Common Issues

## Gazebo Installation and Setup Issues

### Gazebo Not Found

**Issue**: Gazebo commands not found or Gazebo fails to launch
- **Symptoms**: `gz: command not found` or `gazebo: command not found`
- **Solutions**:
  1. Verify Gazebo installation: `apt list --installed | grep gazebo`
  2. Install Gazebo if missing: `sudo apt install ros-humble-gazebo-ros-pkgs`
  3. Check if Gazebo environment is sourced: `source /usr/share/gazebo/setup.sh`
  4. Verify Gazebo version compatibility with ROS 2 Humble

### Gazebo Fails to Launch

**Issue**: Gazebo crashes or fails to start
- **Symptoms**: Gazebo exits immediately or displays error messages
- **Solutions**:
  1. Check graphics drivers and X11 forwarding if using remote access
  2. Try running with software rendering: `export LIBGL_ALWAYS_SOFTWARE=1`
  3. Verify available memory and graphics resources
  4. Check for conflicting processes: `ps aux | grep gazebo`

## Physics Engine Issues

### Physics Instability

**Issue**: Objects jitter, explode, or behave unrealistically
- **Symptoms**: Robot parts flying apart, objects penetrating each other, simulation instability
- **Solutions**:
  1. Reduce time step: `<max_step_size>0.001</max_step_size>` or smaller
  2. Increase update rate: `<real_time_update_rate>2000</real_time_update_rate>`
  3. Check mass and inertia values in URDF (should be physically realistic)
  4. Verify joint limits and effort/velocity parameters

### Performance Issues

**Issue**: Slow simulation or low frame rate
- **Symptoms**: Simulation running below real-time factor, dropped frames
- **Solutions**:
  1. Increase time step (but not too much): `<max_step_size>0.01</max_step_size>`
  2. Reduce visual complexity (disable unnecessary visuals)
  3. Use simpler collision meshes (boxes instead of complex shapes)
  4. Check system resources (CPU, GPU, memory)

## Robot Spawning Issues

### Robot Not Spawning

**Issue**: Robot model fails to appear in simulation
- **Symptoms**: Spawn command succeeds but no robot visible in Gazebo
- **Solutions**:
  1. Verify URDF file syntax: `check_urdf /path/to/robot.urdf`
  2. Check for missing mesh files or textures
  3. Verify robot_description parameter is set correctly
  4. Use verbose spawn command: `ros2 run gazebo_ros spawn_entity.py -entity robot -file robot.urdf -x 0 -y 0 -z 0.5 -verbose`

### Robot Falls Through Ground

**Issue**: Robot falls through the ground plane
- **Symptoms**: Robot falls through world and disappears
- **Solutions**:
  1. Check robot initial position (ensure it's above ground)
  2. Verify ground plane exists and is static
  3. Check robot collision properties and mass
  4. Verify physics engine is running (real_time_factor > 0)

## Sensor Simulation Issues

### Sensor Not Publishing Data

**Issue**: Sensor topics exist but no data is published
- **Symptoms**: Topic shows as available but no messages received
- **Solutions**:
  1. Verify sensor plugin is correctly configured in URDF
  2. Check sensor topic remapping in plugin configuration
  3. Ensure sensor link has proper collision/visual properties
  4. Verify Gazebo simulation is running (not paused)

### Camera Images Black or Distorted

**Issue**: Camera sensor produces black images or distorted visuals
- **Symptoms**: All black images, incorrect colors, or geometric distortions
- **Solutions**:
  1. Check camera parameters (horizontal_fov, image dimensions)
  2. Verify camera link position relative to other objects
  3. Check lighting in the world (ensure light sources exist)
  4. Verify camera plugin configuration and topic name

### LIDAR Returns No Data

**Issue**: LIDAR sensor returns empty or all-infinity ranges
- **Symptoms**: LaserScan messages with all infinity values or no returns
- **Solutions**:
  1. Verify LIDAR link position (ensure it's not inside other objects)
  2. Check scan parameters (min/max angles, range limits)
  3. Ensure there are objects within sensor range
  4. Verify ray sensor plugin configuration

## ROS Integration Issues

### Topic Connection Problems

**Issue**: ROS nodes cannot connect to Gazebo sensor topics
- **Symptoms**: No data received from sensor topics despite Gazebo running
- **Solutions**:
  1. Check topic names match between Gazebo plugin and subscriber
  2. Verify ROS namespace usage if applicable
  3. Check if robot_state_publisher is running
  4. Use `ros2 topic list` and `ros2 topic info` to verify topic availability

### TF Frame Issues

**Issue**: TF transforms not available or incorrect
- **Symptoms**: Robot frames not visible in RViz, transform lookup failures
- **Solutions**:
  1. Verify robot_state_publisher is running: `ros2 run robot_state_publisher robot_state_publisher`
  2. Check URDF joint definitions and parent-child relationships
  3. Ensure joint_state_publisher is publishing joint states
  4. Use `ros2 run tf2_tools view_frames` to visualize TF tree

## Unity Integration Issues

### Unity-ROS Connection Failures

**Issue**: Unity cannot connect to ROS bridge
- **Symptoms**: Connection timeouts, no data exchange between Unity and ROS
- **Solutions**:
  1. Verify ROS bridge server is running: `ros2 run rosbridge_server rosbridge_websocket`
  2. Check IP addresses and ports in Unity ROS settings
  3. Ensure firewall is not blocking connection
  4. Verify message type compatibility between Unity and ROS

### Synchronization Problems

**Issue**: Unity visualization not synchronized with Gazebo simulation
- **Symptoms**: Robot position/pose differs between Gazebo and Unity
- **Solutions**:
  1. Check if both systems are publishing to the same topics
  2. Verify coordinate system alignment between Gazebo and Unity
  3. Ensure consistent update rates between systems
  4. Use the same robot state publisher for both visualization systems

## Common Configuration Issues

### URDF Errors

**Issue**: URDF parsing errors or invalid robot models
- **Symptoms**: Error messages during URDF parsing, robot not loading
- **Solutions**:
  1. Validate URDF syntax: `check_urdf my_robot.urdf`
  2. Check for duplicate joint/link names
  3. Verify all required elements (mass, inertia for links)
  4. Ensure proper file paths for meshes and materials

### Inertia Calculation Problems

**Issue**: Robot behaves unrealistically due to incorrect inertia values
- **Symptoms**: Robot tips over easily, unrealistic motion, physics instability
- **Solutions**:
  1. Calculate proper inertia tensors for geometric shapes
  2. Use CAD software to calculate real inertia values
  3. Approximate with simple geometric shapes: `I = 1/12 * m * (w² + h²)` for boxes
  4. Use online calculators for complex shapes

## Performance Optimization

### Reducing Simulation Load

**Issue**: High CPU/GPU usage causing slow simulation
- **Symptoms**: Low real-time factor, dropped frames, system slowdown
- **Solutions**:
  1. Reduce visual complexity (simplify meshes, reduce texture resolution)
  2. Lower physics update rate if precision allows
  3. Use simpler collision meshes (convex hulls instead of detailed meshes)
  4. Disable unnecessary sensors or reduce their update rates

### Memory Management

**Issue**: High memory usage leading to system instability
- **Symptoms**: System slowdown, out of memory errors, Gazebo crashes
- **Solutions**:
  1. Use compressed textures and meshes
  2. Limit history depth for topics with high data rates
  3. Close unnecessary Gazebo models or worlds
  4. Monitor memory usage with system tools

## Debugging Strategies

### Using Gazebo Tools

**Check physics properties**:
```bash
# Launch Gazebo with verbose output
gz sim -r -v 4 my_world.sdf

# Check model properties
gz model --info -m my_robot
```

**Debugging with ROS tools**:
```bash
# Monitor sensor topics
ros2 topic echo /camera/image_raw --field header.stamp

# Check TF tree
ros2 run tf2_tools view_frames

# Monitor system performance
ros2 run topic_tools relay /diagnostics /filtered_diagnostics
```

### Visualization Debugging

**In RViz**:
1. Add TF display to visualize coordinate frames
2. Add RobotModel display to verify URDF structure
3. Add Image display to verify camera feeds
4. Add LaserScan display to verify LIDAR data

**In Gazebo**:
1. Enable wireframe mode to see collision meshes
2. Use contact visualization to see collision points
3. Enable joint visualization to verify joint limits
4. Use the model inspector to check properties

## Key Concepts

- **Real-time Factor**: Ratio of simulation time to real time (1.0 = real-time)
- **Physics Update Rate**: Frequency at which physics calculations are performed
- **TF Frames**: Coordinate frames used for spatial relationships in ROS
- **Collision Meshes**: Simplified geometry used for physics calculations
- **Visual Meshes**: Detailed geometry used for rendering
- **Sensor Noise**: Artificial noise added to make simulation more realistic

## Practical Exercises

1. Create a diagnostic tool that checks common simulation setup issues
2. Implement a performance monitoring node for simulation metrics
3. Build a validation script that checks URDF files for common errors
4. Develop a debugging launch file with visualization tools

## Common Failure Modes

- **Physics Instability**: Incorrect time steps or parameters causing simulation to explode
- **Resource Exhaustion**: High-resolution sensors or complex models causing performance issues
- **Coordinate System Mismatches**: Different coordinate systems causing alignment problems
- **Plugin Configuration Errors**: Improperly configured Gazebo plugins causing failures
- **Network Connection Issues**: ROS bridge or TCP connection problems in distributed simulation
- **Sensor Noise Mismatches**: Simulated sensor noise not matching real-world characteristics
- **Model Loading Failures**: Missing files or incorrect URDF syntax preventing robot loading
- **Timing Synchronization**: Different update rates causing desynchronization between systems