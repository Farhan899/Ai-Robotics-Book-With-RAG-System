import React from 'react';
import { createRoot } from 'react-dom/client';
import ChatWidget from './ChatWidget';

// Create a container for the chat widget
function loadChatWidget() {
  // Create a container div for the widget
  const container = document.createElement('div');
  container.id = 'chat-widget-container';
  document.body.appendChild(container);

  // Render the chat widget
  const root = createRoot(container);
  root.render(<ChatWidget />);
}

// Execute after the DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadChatWidget);
} else {
  loadChatWidget();
}