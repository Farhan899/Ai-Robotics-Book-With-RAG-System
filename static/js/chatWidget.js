// Check if React and ReactDOM are available before loading the chat widget
function loadChatWidget() {
  // Create a container for the chat widget
  const container = document.createElement('div');
  container.id = 'chat-widget-container';
  document.body.appendChild(container);

  // Dynamically load React and ReactDOM if not already present
  if (typeof React === 'undefined' || typeof ReactDOM === 'undefined') {
    // This is a simplified version - in production, you'd want to handle this more robustly
    console.warn('React or ReactDOM not found. The chat widget requires React to function.');
    return;
  }

  // Import the ChatWidget component
  import('/src/components/ChatWidget/chatWidgetLoader.js')
    .then(module => {
      if (module.default) {
        module.default();
      }
    })
    .catch(err => {
      console.error('Error loading chat widget:', err);
    });
}

// Execute after the DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadChatWidget);
} else {
  loadChatWidget();
}