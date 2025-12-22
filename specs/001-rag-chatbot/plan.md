# Implementation Plan: RAG Chatbot for Robotics Book

**Branch**: `001-rag-chatbot` | **Date**: 2025-12-22 | **Spec**: [link to spec.md](spec.md)
**Input**: Feature specification from `/specs/001-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature implements a Retrieval-Augmented Generation (RAG) chatbot for a robotics book. The system will first query the Qdrant vector database for semantically similar content before generating responses using the OpenRouter API with the z-ai/glm-4.5-air:free model. It handles user-selected text analysis and provides graceful fallback mechanisms when database access fails.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: OpenAI SDK, FastAPI, Qdrant client library, OpenRouter API
**Storage**: Qdrant vector database for robotics content, with potential local caching
**Testing**: pytest for unit tests, integration tests for API calls and RAG pipeline
**Target Platform**: Linux server (backend API)
**Project Type**: Backend API service
**Performance Goals**: 95% of user queries receive responses within 5 seconds
**Constraints**: Must handle Qdrant API failures gracefully, avoid exposing API keys
**Scale/Scope**: Single application supporting multiple concurrent users asking questions about robotics content

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

This implementation plan aligns with the AI Robotics Book RAG Chatbot Constitution:

- I. Retrieval-Augmented Generation (RAG) First: Plan includes querying the Qdrant vector database as the first step before generating responses
- II. Book Content Priority: Plan ensures retrieved content is directly cited with "Book Section: [metadata if available]" notation
- III. Test-First (NON-NEGOTIABLE): Plan specifies comprehensive testing with pytest for unit tests and integration tests
- IV. API Integration Testing: Plan covers integration testing for Qdrant connectivity, OpenRouter API calls, and query processing pipeline
- V. Observability and Transparency: Plan includes proper logging and response source indication
- VI. Ethical AI Response: Plan ensures responses are ethical, non-harmful, and fact-based

All constitutional principles are satisfied by this implementation approach.

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── query.py        # Query model and validation
│   │   ├── response.py     # Response model and validation
│   │   └── document.py     # Document model for retrieved content
│   ├── services/
│   │   ├── rag_service.py  # Main RAG service orchestrating retrieval and generation
│   │   ├── qdrant_service.py # Service for interacting with Qdrant vector database
│   │   └── llm_service.py  # Service for OpenRouter API calls
│   ├── api/
│   │   ├── main.py         # Main FastAPI app and routing
│   │   ├── chat_router.py  # Chat-related endpoints
│   │   └── health_router.py # Health and status endpoints
│   └── core/
│       ├── config.py       # Configuration management
│       ├── logging.py      # Logging setup
│       └── middleware.py   # Request/response middleware
└── tests/
    ├── unit/
    │   ├── test_rag_service.py
    │   ├── test_qdrant_service.py
    │   └── test_llm_service.py
    ├── integration/
    │   ├── test_chat_endpoints.py
    │   └── test_retrieval_flow.py
    └── contract/
        └── test_api_contracts.py
```

**Structure Decision**: Backend API service using FastAPI with clear separation of concerns. The structure follows the single project approach with logical separation of models, services, API endpoints, and core utilities. The testing structure includes unit, integration, and contract tests as required by the constitution.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (No violations identified) | | |