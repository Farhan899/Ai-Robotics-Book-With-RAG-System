<!--
Sync Impact Report:
- Version change: N/A → 1.0.0 (Initial constitution for AI Robotics Book RAG Chatbot project)
- Modified principles: N/A (New project principles)
- Added sections: All principles and sections (New project constitution)
- Removed sections: N/A
- Templates requiring updates: N/A (No existing templates to update for new project)
- Follow-up TODOs: None
-->
# AI Robotics Book RAG Chatbot Constitution

## Core Principles

### I. Retrieval-Augmented Generation (RAG) First
Responses MUST be grounded in retrieved content from the Qdrant vector database containing robotics_docs collection; All answers should first query the database for relevant context before generating a response; When no relevant data is found, explicitly state this limitation and provide general knowledge while noting it's not from the book.

### II. Book Content Priority
Information from the robotics book takes precedence over general knowledge; Retrieved content must be directly cited with "Book Section: [metadata if available]" notation; When user provides text selection, responses must be solely based on that text regardless of database retrieval.

### III. Test-First (NON-NEGOTIABLE)
All RAG functionality must have comprehensive tests covering retrieval accuracy, response generation quality, and error handling; Tests must be written before implementation with clear expected outcomes; Red-Green-Refactor cycle strictly enforced for all features.

### IV. API Integration Testing
Focus areas requiring integration tests: Qdrant database connectivity, OpenRouter API calls, Query processing pipeline, Response formatting and citations; Mock services must simulate external dependencies for reliable testing.

### V. Observability and Transparency
All queries and responses must be logged for debugging and quality assurance; API calls to external services must include proper error handling and monitoring; System must clearly indicate query type (e.g., "Based on book retrieval:" or "Using selected text:").

### VI. Ethical AI Response


AI responses must adhere to ethical guidelines preventing harmful, misleading, or biased content; User privacy must be respected with no data storage beyond session requirements; Responses must be fact-based and neutral, especially for sensitive robotics topics.

## Performance and Quality Standards

Technology stack requirements: OpenAI Agents SDKs, ChatKit SDKs, FastAPI for backend, OpenRouter API with z-ai/glm-4.5-air:free model; Compliance standards: No storing of personal data beyond session, adherence to ethical AI guidelines; Deployment policies: Secure API key management, rate limiting for API calls, appropriate error handling.

## Development Workflow and Review Process

Code review requirements: All pull requests must verify compliance with ethical AI guidelines and proper error handling; Testing gates: All RAG functionality must pass both unit and integration tests before merging; Deployment approval process: Manual verification of response quality and adherence to constitutional principles required.

## Governance

Constitution supersedes all other practices for the AI Robotics Book RAG Chatbot; Amendments require documentation of changes and formal approval; Quality reviews must verify ongoing compliance with all constitutional principles.

All PRs/reviews must verify constitutional compliance; Complexity must be justified with clear value proposition; Use this constitution for development guidance.

**Version**: 1.0.0 | **Ratified**: 2025-06-13 | **Last Amended**: 2025-12-22
