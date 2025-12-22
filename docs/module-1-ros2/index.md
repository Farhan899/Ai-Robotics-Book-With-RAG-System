---
sidebar_position: 1
---

# Module 1: The Robotic Nervous System (ROS 2)

## Overview

Welcome to Module 1: The Robotic Nervous System (ROS 2). In this module, you'll learn about the Robot Operating System 2 (ROS 2), which serves as the middleware that connects all components of a robotic system. ROS 2 provides the communication infrastructure that allows different parts of a robot to work together seamlessly.

### What You'll Learn

In this module, you will:
- Understand the architecture of ROS 2 and its underlying DDS middleware
- Learn how to create and manage nodes, topics, services, and actions
- Explore Quality of Service (QoS) policies and their impact on real-time constraints
- Bridge Python agents to ROS controllers using `rclpy`
- Work with URDF for humanoid robots, including links, joints, kinematic chains, and collision vs visual meshes

### Prerequisites

Before starting this module, ensure you have:
- Completed the Introduction module
- Installed ROS 2 Humble Hawksbill
- Basic Python programming knowledge
- Understanding of fundamental robotics concepts

## ROS 2 Architecture

ROS 2 is a flexible framework for writing robot software that provides services designed for a heterogeneous computer cluster. Unlike ROS 1, which used a centralized master approach, ROS 2 uses a distributed architecture based on the Data Distribution Service (DDS) middleware.

### Key Components

1. **Nodes**: Processes that perform computation. Nodes are the fundamental building blocks of a ROS 2 program.
2. **Topics**: Named buses over which nodes exchange messages.
3. **Services**: Synchronous request/response communication between nodes.
4. **Actions**: Asynchronous request/goal-based communication with feedback and status updates.
5. **Parameters**: Configuration values that can be accessed and changed at runtime.

## Quality of Service (QoS) Policies

ROS 2 introduces Quality of Service (QoS) policies that allow you to fine-tune communication behavior. QoS policies include:
- Reliability: Whether messages are guaranteed to be delivered
- Durability: Whether late-joining subscribers get old messages
- History: How many messages to store for late-joining subscribers
- Deadline: The maximum time between data updates
- Lifespan: How long messages are kept around before being dropped

## Next Steps

In the next sections of this module, we'll dive deeper into each of these concepts with hands-on exercises and practical examples. You'll learn how to implement these concepts in code and see how they work in practice.

## Key Concepts

- **ROS 2 (Robot Operating System 2)**: A flexible framework for writing robot software that provides services designed for a heterogeneous computer cluster
- **DDS (Data Distribution Service)**: A middleware protocol that enables scalable, real-time, dependable, distributed data exchange
- **Nodes**: Processes that perform computation; the fundamental building blocks of a ROS 2 program
- **Topics**: Named buses over which nodes exchange messages using publish-subscribe pattern
- **Services**: Synchronous request/response communication between nodes
- **Actions**: Asynchronous request/goal-based communication with feedback and status updates
- **QoS (Quality of Service)**: Policies that allow fine-tuning of communication behavior

## Practical Exercises

1. Set up a ROS 2 development environment with Humble Hawksbill
2. Create and run your first ROS 2 publisher and subscriber nodes
3. Implement a simple service for inter-node communication
4. Build a basic URDF model of a humanoid robot
5. Visualize your robot in RViz

## Common Failure Modes

- **Environment Setup Issues**: Problems with sourcing ROS 2 setup files or PATH configuration
- **Node Communication Issues**: Nodes failing to communicate due to topic name mismatches or QoS policy conflicts
- **URDF Parsing Errors**: XML syntax errors preventing robot model loading
- **Resource Management**: Memory leaks from improper node cleanup
- **Network Configuration**: Multi-machine communication issues due to firewall or domain settings

## Next Module

Continue to [Module 2: The Digital Twin (Gazebo & Unity)](/docs/module-2-digital-twin) to learn about physics simulation and environment building.