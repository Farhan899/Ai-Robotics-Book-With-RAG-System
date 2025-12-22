# Data Model: RAG Chatbot for Robotics Book

## Overview
This document defines the data models for the Retrieval-Augmented Generation (RAG) chatbot system. It details the entities, their attributes, relationships, and validation rules based on the feature requirements.

## Core Entities

### Query
Represents a user's input to the system.

**Fields**:
- `id` (str): Unique identifier for the query
- `text` (str): The actual question or text provided by the user
- `timestamp` (datetime): When the query was received
- `user_id` (str, optional): Identifier for the user (if tracking is enabled)
- `selected_text` (str, optional): Specific text provided by user for analysis

**Validation Rules**:
- `text` must be between 1 and 2000 characters
- `selected_text` cannot exceed 5000 characters if provided

**State Transitions**: None (queries are immutable once received)

### RetrievedContext
Represents the context retrieved from the Qdrant database.

**Fields**:
- `id` (str): Unique identifier for the context
- `content` (str): The retrieved text content from the robotics book
- `metadata` (dict): Additional metadata about the source (e.g., chapter, page, section)
- `similarity_score` (float): The similarity score from the vector search
- `source_document_id` (str): Identifier for the original document

**Validation Rules**:
- `content` must not be empty
- `similarity_score` must be between 0 and 1
- `metadata` must include at least a source identifier

### Response
Represents the system's response to the user's query.

**Fields**:
- `id` (str): Unique identifier for the response
- `query_id` (str): Reference to the original query
- `content` (str): The generated response text
- `citations` (list[str]): List of citations in the format "Book Section: [metadata]"
- `response_type` (enum): Type of response (e.g., "from_book", "fallback_general", "fallback_unavailable")
- `timestamp` (datetime): When the response was generated

**Validation Rules**:
- `content` must not be empty
- `response_type` must be one of the defined enum values
- If `response_type` is "from_book", `citations` must not be empty

### Document
Represents a document in the Qdrant vector database.

**Fields**:
- `id` (str): Unique identifier for the document
- `content` (str): The text content of the document
- `metadata` (dict): Metadata including source, section, chapter, etc.
- `embedding` (list[float]): The vector embedding of the content
- `created_at` (datetime): When the document was indexed

**Validation Rules**:
- `content` must not be empty
- `embedding` must have the correct dimension for the Qdrant collection
- `metadata` must include required fields for citation

## Relationships

```
Query (1) -- (0..*) RetrievedContext
Query (1) -- (1) Response
RetrievedContext (0..*) -- (1) Document
```

## Validation Rules Summary

1. **Query Validation**: 
   - Text field length between 1-2000 characters
   - Selected text length not exceeding 5000 characters

2. **Response Validation**:
   - Content must not be empty
   - Appropriate citations for book-based responses
   - Valid response type enumeration

3. **Context Validation**:
   - Non-empty content
   - Valid similarity scores between 0-1
   - Required metadata fields

4. **Document Validation**:
   - Non-empty content
   - Valid vector embeddings
   - Required metadata for citations

## State Transitions

- **Query**: Immutable after creation
- **RetrievedContext**: Immutable after creation  
- **Response**: Immutable after creation
- **Document**: Immutable after creation and indexing

## Notes

- All entities follow an immutable pattern after creation to maintain consistency
- The system does not store user data beyond the session requirements as per ethical guidelines
- Metadata structure for citations should follow the format "Book Section: [relevant metadata if available]" as required by the specification