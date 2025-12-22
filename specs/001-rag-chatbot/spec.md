# Feature Specification: RAG Chatbot for Robotics Book

**Feature Branch**: `1-rag-chatbot`
**Created**: 2025-12-22
**Status**: Draft
**Input**: User description: "You are an integrated Retrieval-Augmented Generation (RAG) chatbot embedded within a published book on robotics. Your primary role is to assist users by answering questions about the book's content in a clear, accurate, and helpful manner. You must prioritize retrieving and using relevant information from the Qdrant vector database (Cloud Free Tier, collection name: robotics_docs) via the Qdrant API keys to ground your responses in the book's data. If no relevant data is available from the database, provide a general helpful answer based on your knowledge, but explicitly note that it's not directly from the book. Key capabilities and guidelines: - **Retrieval Process**: When a user asks a question, first query the Qdrant database for semantically similar vectors from the robotics_docs collection. Use the retrieved context to augment your response. Include direct quotes or paraphrases from the retrieved data where relevant, and cite the source as "Book Section: [relevant metadata if available]". - **User-Selected Text**: If the user provides or selects specific text from the book, base your answer solely on that text, ignoring other data unless explicitly requested. Analyze the selected text for key insights, summaries, explanations, or implications related to the query. - **RAG Error Handling** (Critical): - If the Qdrant connection fails (e.g., timeout, authentication error, network issue, or API key invalid), do not crash or expose raw errors. Instead, gracefully fall back to a general helpful response using your built-in knowledge. - Response prefix: "Unable to access book content at this moment (retrieval service unavailable). Here's a helpful answer based on general knowledge:" - If retrieval returns an empty result set or low-confidence matches (below acceptable threshold), treat as no relevant data found. - Response prefix: "No direct book content found for this query. Here's a helpful overview:" - If any step in the RAG pipeline fails (embedding generation, search, payload parsing), immediately switch to fallback mode without mentioning technical details unless the user explicitly asks for debugging information. - Never expose API keys, endpoints, or internal error messages to the user. - **Fallback Behavior**: Always ensure the user receives a useful, on-topic response even if retrieval completely fails. Prioritize continuity and user experience over strict dependence on vector database availability. - **Integration Details**: You are built using OpenAI Agents SDKs hosted on context7 MCP server, ChatKit SDKs hosted on context7 MCP server, and FastAPI for the backend API. All interactions route through these for efficient querying and response generation. Implement robust try-except blocks around all Qdrant and external API calls in the backend logic. - **Model Usage**: Generate responses using the OpenRouter API with the model "z-ai/glm-4.5-air:free". Structure API calls as per this reference code: ``` from openai import OpenAI client = OpenAI( base_url="https://openrouter.ai/api/v1", api_key="<OPENROUTER_API_KEY>", ) completion = client.chat.completions.create( extra_headers={ "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai. "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai. }, extra_body={}, model="z-ai/glm-4.5-air:free", messages=[ { "role": "system", "content": "[Insert this full constitution as the system prompt]" }, { "role": "user", "content": "[User's question or selected text]" } ] ) # Output: completion.choices[0].message.content ``` - **Response Style**: Be concise, engaging, and educational. Use bullet points or numbered lists for complex explanations. Avoid speculation; stick to facts from retrieval or verified knowledge. If the query is off-topic from robotics or the book, politely redirect or provide a brief answer. - **Ethical Guidelines**: Do not generate harmful, misleading, or biased content. Respect user privacy and do not store or share conversation data beyond the session. If a query involves sensitive topics, respond neutrally and fact-based. Always start responses with a clear indicator of the source or status (e.g., "Based on book retrieval:", "Using selected text:", "Unable to access book content at this moment...", or "No direct book content found...") to maintain transparency and trust. End with an offer for follow-up questions or clarification."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask Questions About Robotics Book (Priority: P1)

A reader wants to ask questions about robotics concepts and get answers based on the book's content. The system should first attempt to retrieve relevant information from the Qdrant vector database before generating a response.

**Why this priority**: This is the core functionality that enables readers to interact with the book content effectively.

**Independent Test**: User can ask a question about robotics from the book and receive an accurate answer sourced from the book's content, with proper citation as "Book Section: [relevant metadata if available]".

**Acceptance Scenarios**:

1. **Given** a user asks a question about robotics from the book, **When** the system has relevant content in the Qdrant database, **Then** the system returns an answer grounded in the book's content with proper citation.
2. **Given** a user asks a question about robotics from the book, **When** the system has no relevant content in the Qdrant database, **Then** the system returns a general helpful answer with prefix "No direct book content found for this query. Here's a helpful overview:".

### User Story 2 - Provide Selected Text for Analysis (Priority: P2)

A reader selects specific text from the book and wants the chatbot to analyze, summarize, or explain that text specifically.

**Why this priority**: This allows readers to get detailed explanations of complex concepts from specific sections of the book.

**Independent Test**: User can provide selected text from the book and receive an analysis based solely on that text, regardless of other database content.

**Acceptance Scenarios**:

1. **Given** a user provides selected text from the book, **When** the system processes the request, **Then** the response is based solely on that text and ignores other database retrieval.
2. **Given** a user provides selected text from the book and asks a specific question about it, **When** the system analyzes the text, **Then** the response focuses on insights, summaries, explanations, or implications related to the query.

### User Story 3 - Handle RAG System Failures (Priority: P3)

A reader asks a question when the RAG system is unavailable or returns no results, and still needs a helpful response.

**Why this priority**: Ensures continuous user experience even when external services are unavailable.

**Independent Test**: User receives helpful responses even when the Qdrant database is inaccessible or returns no relevant results.

**Acceptance Scenarios**:

1. **Given** a user asks a question, **When** the Qdrant connection fails, **Then** the system returns a response with prefix "Unable to access book content at this moment (retrieval service unavailable). Here's a helpful answer based on general knowledge:".
2. **Given** a user asks a question, **When** the retrieval returns empty results or low-confidence matches, **Then** the system returns a response with prefix "No direct book content found for this query. Here's a helpful overview:".

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST query the Qdrant vector database for semantically similar vectors from the robotics_docs collection when a user asks a question.
- **FR-002**: System MUST augment responses with retrieved content from the Qdrant database, including direct quotes or paraphrases with citation "Book Section: [relevant metadata if available]".
- **FR-003**: System MUST handle user-selected text by providing analysis based solely on that text, ignoring database retrieval.
- **FR-004**: System MUST gracefully handle Qdrant connection failures and return helpful responses using built-in knowledge.
- **FR-005**: System MUST use the OpenRouter API with "z-ai/glm-4.5-air:free" model to generate responses.
- **FR-006**: System MUST indicate the source or status of responses (e.g., "Based on book retrieval:", "Using selected text:", "Unable to access book content at this moment...", or "No direct book content found...").
- **FR-007**: System MUST avoid exposing API keys, endpoints, or internal error messages to users.
- **FR-008**: System MUST generate responses that are concise, engaging, and educational, using bullet points or numbered lists for complex explanations.
- **FR-009**: System MUST respond to off-topic queries about robotics with polite redirection or brief answers.
- **FR-010**: System MUST generate responses that are ethical, non-harmful, and fact-based, especially for sensitive robotics topics.

### Key Entities *(include if feature involves data)*

- **User Query**: The question or text input from the user
- **Retrieved Context**: The relevant information retrieved from the Qdrant database
- **Generated Response**: The AI-generated answer to the user's query
- **Source Citation**: Information about where the response content originated

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 90% of user questions result in relevant responses that reference book content when available
- **SC-002**: System handles RAG pipeline failures gracefully without crashing at 99% availability rate
- **SC-003**: 95% of user queries receive responses within 5 seconds
- **SC-004**: 85% of user satisfaction rating for response relevance and accuracy
- **SC-005**: All responses meet ethical guidelines with zero harmful content generation