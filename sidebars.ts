import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar structure for the AI/Spec-Driven Humanoid Robotics book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
      link: {
        type: 'doc',
        id: 'intro',
      },
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/index',
        'module-1-ros2/architecture',
        'module-1-ros2/hands-on',
        'module-1-ros2/code-examples',
        'module-1-ros2/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/index',
        'module-2-digital-twin/architecture',
        'module-2-digital-twin/hands-on',
        'module-2-digital-twin/code-examples',
        'module-2-digital-twin/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-3-ai-robot-brain/index',
        'module-3-ai-robot-brain/architecture',
        'module-3-ai-robot-brain/hands-on',
        'module-3-ai-robot-brain/code-examples',
        'module-3-ai-robot-brain/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/index',
        'module-4-vla/architecture',
        'module-4-vla/hands-on',
        'module-4-vla/code-examples',
        'module-4-vla/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project: The Autonomous Humanoid',
      items: [
        'capstone/index',
        'capstone/architecture',
        'capstone/hands-on',
        'capstone/code-examples',
        'capstone/troubleshooting',
      ],
    },
  ],
};

export default sidebars;
