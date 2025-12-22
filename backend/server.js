const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5001;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage for conversation history (in production, use a proper database)
let conversationHistory = {};

// Sample robotics knowledge base for context
const knowledgeBase = [
  "ROS 2 (Robot Operating System 2) is flexible framework for writing robot software.",
  "ROS 2 uses DDS (Data Distribution Service) for communication between nodes.",
  "Gazebo is a robot simulation environment that provides physics simulation.",
  "Navigation in ROS involves global path planning, local path planning, and localization.",
  "SLAM (Simultaneous Localization and Mapping) allows robots to map unknown environments.",
  "Computer vision in robotics includes object detection, tracking, and SLAM.",
  "Robotic manipulation involves controlling robot arms to interact with objects.",
  "A digital twin is a virtual representation of a physical system or process.",
  "Vision-Language-Action (VLA) models connect visual perception with language and actions.",
  "NVIDIA Isaac provides simulation and AI tools for robotics development."
];

// Function to generate AI response based on knowledge base
const generateResponse = (message, context = '') => {
  const lowerMsg = message.toLowerCase();
  
  // Check for specific robotics topics
  if (lowerMsg.includes('hello') || lowerMsg.includes('hi') || lowerMsg.includes('hey')) {
    return "Hello! I'm your AI assistant for the AI Robotics Book. I can help answer questions about ROS 2, digital twins, AI perception systems, and other robotics concepts. What would you like to know?";
  } else if (lowerMsg.includes('ros') || lowerMsg.includes('robot operating system')) {
    return "ROS (Robot Operating System) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. In ROS 2, which is the current version, they use DDS (Data Distribution Service) for communication between nodes.";
  } else if (lowerMsg.includes('gazebo') || lowerMsg.includes('simulation')) {
    return "Gazebo is a robot simulation environment that provides physics simulation, sensor simulation, and realistic environments for testing robotic systems before deployment. It integrates well with ROS and allows you to test your robot algorithms in a safe, virtual environment before running them on real hardware.";
  } else if (lowerMsg.includes('navigation') || lowerMsg.includes('path') || lowerMsg.includes('move')) {
    return "Robot navigation typically involves several components: global path planning (finding the optimal path from start to goal), local path planning (avoiding obstacles while following the global path), localization (determining the robot's position), and mapping (creating a representation of the environment). In ROS, the Navigation2 stack provides these capabilities.";
  } else if (lowerMsg.includes('computer vision') || lowerMsg.includes('vision') || lowerMsg.includes('opencv')) {
    return "Computer vision in robotics involves processing camera images to understand the environment. This can include object detection, tracking, SLAM (Simultaneous Localization and Mapping), and depth estimation. Libraries like OpenCV and deep learning frameworks like PyTorch/TensorFlow are commonly used for vision tasks in robotics.";
  } else if (lowerMsg.includes('manipulation') || lowerMsg.includes('arm') || lowerMsg.includes('grasp')) {
    return "Robotic manipulation involves controlling robot arms to interact with objects. This includes inverse kinematics to calculate joint angles needed to reach a position, grasp planning to determine how to grip objects, and trajectory planning to move the arm smoothly and avoid obstacles.";
  } else if (lowerMsg.includes('slam')) {
    return "SLAM (Simultaneous Localization and Mapping) is a technique that allows robots to build a map of an unknown environment while simultaneously keeping track of their location within that environment. Common approaches include EKF SLAM, FastSLAM, and graph-based SLAM. In ROS, packages like Cartographer and RTAB-Map provide SLAM capabilities.";
  } else if (lowerMsg.includes('digital twin')) {
    return "A digital twin is a virtual representation of a physical system or process. In robotics, digital twins allow you to simulate, analyze, and optimize robot behavior before implementing it in the real world. This helps reduce development time, test scenarios safely, and improve overall system performance.";
  } else if (lowerMsg.includes('vision') && lowerMsg.includes('language') && lowerMsg.includes('action')) {
    return "Vision-Language-Action (VLA) models connect visual perception with language understanding and physical actions. These models allow robots to interpret natural language commands, understand their visual environment, and execute appropriate actions. Examples include models that can follow instructions like 'Pick up the red block' by perceiving the scene, understanding the command, and executing the action.";
  } else {
    // Find relevant information from knowledge base
    const relevantFacts = knowledgeBase.filter(fact => 
      lowerMsg.split(' ').some(word => fact.toLowerCase().includes(word))
    );
    
    if (relevantFacts.length > 0) {
      return `Based on the AI Robotics Book content: ${relevantFacts[0]}`;
    } else {
      return `I understand you're asking about "${message}". In the context of AI robotics, this could involve several concepts. The AI Robotics Book covers topics such as ROS 2, digital twin environments, AI perception systems, and Vision-Language-Action frameworks. Could you ask about a specific robotics concept for a more detailed explanation?`;
    }
  }
};

// Chat endpoint - now proxies to the RAG backend
app.post('/api/chat', async (req, res) => {
  try {
    const { message, context, sessionId = 'default' } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // Forward the request to the Python RAG backend
    const ragResponse = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: message,
        selected_text: null
      }),
    });

    if (ragResponse.ok) {
      const ragData = await ragResponse.json();

      // Format as the expected reply format
      let formattedReply = ragData.content || 'I received your message and processed it.';

      if (ragData.citations && ragData.citations.length > 0) {
        formattedReply += '\n\nSources:\n' + ragData.citations.map((citation) => `- ${citation}`).join('\n');
      }

      // Store conversation in memory (in production, use a database)
      if (!conversationHistory[sessionId]) {
        conversationHistory[sessionId] = [];
      }

      // Add to conversation history
      conversationHistory[sessionId].push({
        user: message,
        bot: formattedReply,
        timestamp: new Date()
      });

      // Limit conversation history to last 10 exchanges
      if (conversationHistory[sessionId].length > 20) {
        conversationHistory[sessionId] = conversationHistory[sessionId].slice(-20);
      }

      return res.json({
        reply: formattedReply,
        sessionId,
        timestamp: new Date()
      });
    } else {
      // If RAG backend fails, fall back to the local knowledge base
      console.warn('RAG backend unavailable, falling back to local knowledge base');
      const fallbackResponse = generateResponse(message, context);

      // Store conversation in memory (in production, use a database)
      if (!conversationHistory[sessionId]) {
        conversationHistory[sessionId] = [];
      }

      // Add to conversation history
      conversationHistory[sessionId].push({
        user: message,
        bot: fallbackResponse,
        timestamp: new Date()
      });

      // Limit conversation history to last 10 exchanges
      if (conversationHistory[sessionId].length > 20) {
        conversationHistory[sessionId] = conversationHistory[sessionId].slice(-20);
      }

      return res.json({
        reply: fallbackResponse,
        sessionId,
        timestamp: new Date()
      });
    }
  } catch (error) {
    console.error('Error processing chat request:', error);
    res.status(500).json({
      error: 'Internal server error',
      reply: 'I\'m having trouble processing your request. Please try again later.'
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date() });
});

// Serve static files if needed
app.use(express.static(path.join(__dirname, '../build')));

// Catch-all handler for any non-API routes (for SPA)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../build/index.html'));
});

app.listen(PORT, () => {
  console.log(`AI Robotics Chat Server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
  console.log(`Chat endpoint: http://localhost:${PORT}/api/chat`);
});