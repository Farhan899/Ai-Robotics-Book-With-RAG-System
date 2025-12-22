# Tasks: RAG Chatbot for Robotics Book

**Input**: Design documents from `/specs/001-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The feature specification indicates comprehensive testing is required to satisfy the constitutional principles. Test-First approach is mandated by constitution.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume backend API structure based on plan.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan in backend/
- [X] T002 Initialize Python 3.11 project with FastAPI, OpenAI SDK, Qdrant client library, OpenRouter API dependencies
- [X] T003 [P] Configure linting and formatting tools (flake8, black, mypy)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [X] T004 Setup configuration management with environment variables in backend/src/core/config.py
- [X] T005 [P] Configure logging infrastructure in backend/src/core/logging.py
- [X] T006 [P] Setup API routing and middleware structure in backend/src/core/middleware.py
- [X] T007 Create base models/entities that all stories depend on in backend/src/models/
- [X] T008 Configure error handling infrastructure in backend/src/core/middleware.py
- [X] T009 Setup environment configuration management with pydantic models in backend/src/core/config.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Ask Questions About Robotics Book (Priority: P1) 🎯 MVP

**Goal**: Enable users to ask questions about robotics concepts and get answers based on the book's content with proper citations.

**Independent Test**: User can ask a question about robotics from the book and receive an accurate answer sourced from the book's content, with proper citation as "Book Section: [relevant metadata if available]".

### Tests for User Story 1 (Mandatory - Test-First approach per constitution) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T010 [P] [US1] Contract test for /chat endpoint in backend/tests/contract/test_api_contracts.py
- [X] T011 [P] [US1] Integration test for RAG pipeline in backend/tests/integration/test_retrieval_flow.py
- [X] T012 [P] [US1] Unit test for rag_service.py in backend/tests/unit/test_rag_service.py
- [X] T013 [P] [US1] Unit test for qdrant_service.py in backend/tests/unit/test_qdrant_service.py
- [X] T014 [P] [US1] Unit test for llm_service.py in backend/tests/unit/test_llm_service.py

### Implementation for User Story 1

- [X] T015 [P] [US1] Create Query model in backend/src/models/query.py
- [X] T016 [P] [US1] Create Response model in backend/src/models/response.py
- [X] T017 [P] [US1] Create Document model in backend/src/models/document.py
- [X] T018 [US1] Implement rag_service.py with RAG orchestration in backend/src/services/rag_service.py
- [X] T019 [US1] Implement qdrant_service.py for Qdrant interactions in backend/src/services/qdrant_service.py
- [X] T020 [US1] Implement llm_service.py for OpenRouter API calls in backend/src/services/llm_service.py
- [X] T021 [US1] Implement chat_router.py with /chat endpoint in backend/src/api/chat_router.py
- [X] T022 [US1] Add validation and error handling for user story 1
- [X] T023 [US1] Add logging for user story 1 operations
- [X] T024 [US1] Implement response citation logic matching "Book Section: [metadata if available]" format

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Provide Selected Text for Analysis (Priority: P2)

**Goal**: Allow users to provide specific text from the book for focused analysis, with responses based solely on that text.

**Independent Test**: User can provide selected text from the book and receive an analysis based solely on that text, regardless of other database content.

### Tests for User Story 2 (Mandatory - Test-First approach per constitution) ⚠️

- [X] T025 [P] [US2] Contract test for /chat/text-analysis endpoint in backend/tests/contract/test_api_contracts.py
- [X] T026 [P] [US2] Integration test for text analysis pipeline in backend/tests/integration/test_text_analysis.py
- [X] T027 [P] [US2] Unit test for text analysis in rag_service.py in backend/tests/unit/test_rag_service.py

### Implementation for User Story 2

- [X] T028 [P] [US2] Extend Query model to support selected_text in backend/src/models/query.py
- [X] T029 [US2] Update rag_service.py to handle user-selected text differently in backend/src/services/rag_service.py
- [X] T030 [US2] Implement /chat/text-analysis endpoint in backend/src/api/chat_router.py
- [X] T031 [US2] Add validation for selected text length and content in backend/src/models/query.py
- [X] T032 [US2] Ensure selected text analysis ignores database retrieval in backend/src/services/rag_service.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Handle RAG System Failures (Priority: P3)

**Goal**: Ensure users still get helpful responses when the RAG system is unavailable, with graceful fallback mechanisms.

**Independent Test**: User receives helpful responses even when the Qdrant database is inaccessible or returns no relevant results.

### Tests for User Story 3 (Mandatory - Test-First approach per constitution) ⚠️

- [X] T033 [P] [US3] Integration test for Qdrant failure handling in backend/tests/integration/test_retrieval_flow.py
- [X] T034 [P] [US3] Unit test for fallback behavior in rag_service.py in backend/tests/unit/test_rag_service.py
- [X] T035 [P] [US3] Contract test for graceful degradation responses in backend/tests/contract/test_api_contracts.py

### Implementation for User Story 3

- [X] T036 [P] [US3] Implement Qdrant connection error handling in backend/src/services/qdrant_service.py
- [X] T037 [P] [US3] Implement general error handling in rag_service.py in backend/src/services/rag_service.py
- [X] T038 [US3] Add fallback logic for empty Qdrant results in backend/src/services/rag_service.py
- [X] T039 [US3] Implement response prefix for unavailable content in backend/src/services/rag_service.py
- [X] T040 [US3] Add appropriate response_type handling for fallback scenarios in backend/src/models/response.py
- [X] T041 [US3] Ensure no API keys or internal errors are exposed to users in error responses

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T042 [P] Documentation updates in backend/README.md
- [X] T043 Code cleanup and refactoring
- [X] T044 Performance optimization to meet 5-second response time requirement
- [X] T045 [P] Additional unit tests in backend/tests/unit/
- [X] T046 Security hardening including API key management
- [X] T047 Run quickstart.md validation
- [X] T048 Implement health endpoint /health in backend/src/api/health_router.py
- [X] T049 Add comprehensive logging for observability
- [X] T050 Final integration testing of all components

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Builds on US1 components but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Contract test for /chat endpoint in backend/tests/contract/test_api_contracts.py"
Task: "Integration test for RAG pipeline in backend/tests/integration/test_retrieval_flow.py"
Task: "Unit test for rag_service.py in backend/tests/unit/test_rag_service.py"
Task: "Unit test for qdrant_service.py in backend/tests/unit/test_qdrant_service.py"
Task: "Unit test for llm_service.py in backend/tests/unit/test_llm_service.py"

# Launch all models for User Story 1 together:
Task: "Create Query model in backend/src/models/query.py"
Task: "Create Response model in backend/src/models/response.py"
Task: "Create Document model in backend/src/models/document.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence