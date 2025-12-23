import React, { useState, useEffect } from 'react';

// Simple chat widget without using Docusaurus theme hooks to avoid SSR issues
const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, type: 'bot', text: 'Hello! I\'m your AI assistant for the AI Robotics Book. How can I help you today?' }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // For now, we'll use a simple approach without Docusaurus color mode
  // This avoids the context issue during SSR
  const [colorMode, setColorMode] = useState('light');

  // Detect and sync with system preference or site preference if available
  useEffect(() => {
    const detectColorMode = () => {
      if (typeof window !== 'undefined') {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        setColorMode(prefersDark ? 'dark' : 'light');
      }
    };

    detectColorMode();
    
    // Listen for changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => detectColorMode();
    
    mediaQuery.addEventListener('change', handleChange);
    
    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, []);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // Function to send message to AI backend
  const sendToAI = async (message: string) => {
    try {
      // Connect to the backend API server (using relative path to avoid CORS issues)
      // The backend proxy server will forward to the RAG backend
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,  // Node.js proxy expects 'message' field
          context: 'AI Robotics Book'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        return data.reply || 'I received your message and processed it.';
      } else {
        // Log specific error for debugging
        console.error(`API call failed with status: ${response.status} ${response.statusText}`);
        const errorText = await response.text();
        console.error('Error response body:', errorText);

        // Fallback response if API call fails
        return 'I\'m having trouble connecting to the AI service right now. Could you try again?';
      }
    } catch (error) {
      // Log the specific network or fetch error
      console.error('Network or fetch error:', error);

      // Return a relevant response based on common robotics topics
      return getRelevantResponse(message);
    }
  };

  const handleSend = async () => {
    if (!inputText.trim()) return;

    // Add user message
    const userMessage = { id: messages.length + 1, type: 'user', text: inputText };
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    // Get response from AI service
    const botResponse = await sendToAI(inputText);

    const botMessage = {
      id: messages.length + 2,
      type: 'bot',
      text: botResponse
    };

    setMessages(prev => [...prev, botMessage]);
    setIsLoading(false);
  };

  // Helper function to generate relevant responses when API is not available
  const getRelevantResponse = (message: string): string => {
    const lowerMsg = message.toLowerCase();

    // Check for common robotics topics
    if (lowerMsg.includes('ros') || lowerMsg.includes('robot operating system')) {
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
    } else if (lowerMsg.includes('hello') || lowerMsg.includes('hi') || lowerMsg.includes('hey')) {
      return "Hello! I'm your AI assistant for the AI Robotics Book. I can help answer questions about ROS 2, digital twins, AI perception systems, and other robotics concepts. What would you like to know?";
    } else {
      return "That's an interesting question about robotics. In the AI Robotics Book, we cover topics like ROS 2, digital twin environments, AI perception systems, and Vision-Language-Action frameworks. Could you ask about a specific robotics concept you'd like to understand better?";
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-widget">
      {isOpen ? (
        <div className={`chat-window chat-window-${colorMode}`}>
          <div className={`chat-header chat-header-${colorMode}`}>
            <div className="chat-title">AI Robotics Assistant</div>
            <button className="chat-close" onClick={toggleChat}>×</button>
          </div>
          <div className="chat-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.type} message-${colorMode}`}
              >
                {message.text}
              </div>
            ))}
            {isLoading && (
              <div className={`message bot message-${colorMode}`}>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </div>
          <div className={`chat-input-area chat-input-area-${colorMode}`}>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about robotics concepts..."
              className={`chat-input chat-input-${colorMode}`}
              rows={2}
            />
            <button 
              onClick={handleSend} 
              disabled={isLoading || !inputText.trim()}
              className="chat-send-button"
            >
              Send
            </button>
          </div>
        </div>
      ) : null}
      
      <button className={`chat-toggle-button chat-toggle-button-${colorMode}`} onClick={toggleChat}>
        💬
      </button>
      
      <style jsx>{`
        .chat-widget {
          position: fixed;
          bottom: 20px;
          right: 20px;
          z-index: 1000;
        }

        .chat-toggle-button {
          width: 60px;
          height: 60px;
          border-radius: 50%;
          border: none;
          color: white;
          font-size: 24px;
          cursor: pointer;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .chat-toggle-button-light {
          background-color: #4f8bf9;
        }

        .chat-toggle-button-dark {
          background-color: #2c6eaf;
        }

        .chat-window {
          width: 350px;
          height: 500px;
          max-width: 90vw;
          max-height: 70vh;
          display: flex;
          flex-direction: column;
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          margin-bottom: 15px;
        }

        .chat-window-light {
          background-color: #ffffff;
          border: 1px solid #e0e0e0;
        }

        .chat-window-dark {
          background-color: #242526;
          border: 1px solid #444950;
        }

        .chat-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 15px;
        }

        .chat-header-light {
          background-color: #f5f6f7;
          border-bottom: 1px solid #e0e0e0;
        }

        .chat-header-dark {
          background-color: #3a3b3c;
          border-bottom: 1px solid #444950;
        }

        .chat-title {
          font-weight: bold;
          font-size: 16px;
        }

        .chat-close {
          background: none;
          border: none;
          font-size: 24px;
          cursor: pointer;
          color: inherit;
          padding: 0;
          width: 30px;
          height: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 15px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .message {
          max-width: 80%;
          padding: 10px 12px;
          border-radius: 18px;
          position: relative;
        }

        .message-user-light {
          background-color: #4f8bf9;
          color: white;
          align-self: flex-end;
        }

        .message-user-dark {
          background-color: #0d78ea;
          color: white;
          align-self: flex-end;
        }

        .message-bot-light {
          background-color: #f0f2f5;
          color: #333;
          align-self: flex-start;
        }

        .message-bot-dark {
          background-color: #3a3b3c;
          color: #e4e6eb;
          align-self: flex-start;
        }

        .typing-indicator {
          display: flex;
          align-items: center;
          padding: 5px 0;
        }

        .typing-indicator span {
          height: 8px;
          width: 8px;
          background-color: #999;
          border-radius: 50%;
          display: inline-block;
          margin: 0 2px;
          animation: typing 1.4s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(1) {
          animation-delay: -0.32s;
        }

        .typing-indicator span:nth-child(2) {
          animation-delay: -0.16s;
        }

        @keyframes typing {
          0%, 80%, 100% {
            transform: scale(0.8);
            opacity: 0.5;
          }
          40% {
            transform: scale(1);
            opacity: 1;
          }
        }

        .chat-input-area {
          display: flex;
          padding: 10px;
          border-top: 1px solid;
        }

        .chat-input-area-light {
          border-top-color: #e0e0e0;
          background-color: #ffffff;
        }

        .chat-input-area-dark {
          border-top-color: #444950;
          background-color: #242526;
        }

        .chat-input {
          flex: 1;
          border: 1px solid;
          border-radius: 18px;
          padding: 10px 15px;
          resize: none;
          max-height: 100px;
          margin-right: 10px;
          outline: none;
        }

        .chat-input-light {
          background-color: #fff;
          border-color: #e0e0e0;
          color: #333;
        }

        .chat-input-dark {
          background-color: #3a3b3c;
          border-color: #444950;
          color: #e4e6eb;
        }

        .chat-send-button {
          border: none;
          border-radius: 18px;
          padding: 10px 15px;
          background-color: #4f8bf9;
          color: white;
          cursor: pointer;
          align-self: flex-end;
        }

        .chat-send-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        @media (max-width: 480px) {
          .chat-window {
            width: 95vw;
            height: 70vh;
          }
        }
      `}</style>
    </div>
  );
};

export default ChatWidget;