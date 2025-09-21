from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


class DocumentChunk(BaseModel):
    """Represents a chunk of document content"""
    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="The actual text content")
    metadata: Dict[str, Any] = Field(default_factory=lambda: dict, description="Additional metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score for retrieval")

# COMPLETED: Implement the AnswerResponse schema for structured Q&A responses.
# This schema should include fields for the question, answer, sources, confidence, and timestamp.
# Refer to README.md Task 1.1 for detailed field requirements.
class AnswerResponse(BaseModel):
    """Structured response for Q&A tasks"""
    question: str = Field("The original user question")
    answer: str = Field("The generated answer")
    sources: List[str] = Field("List of source document IDs used")
    confidence: float = Field("Confidence score between 0 and 1")
    timestamp: datetime = Field("When the response was generated")

class SummarizationResponse(BaseModel):
    """Structured response for summarization tasks"""
    original_length: int = Field(description="Length of original text")
    summary: str = Field(description="The generated summary")
    key_points: List[str] = Field(description="List of key points extracted")
    document_ids: List[str] = Field(default_factory=lambda: list, description="Documents summarized")
    timestamp: datetime = Field(default_factory=datetime.now)


class CalculationResponse(BaseModel):
    """Structured response for calculation tasks"""
    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Step-by-step explanation")
    units: Optional[str] = Field(default=None, description="Units if applicable")
    timestamp: datetime = Field(default_factory=datetime.now)

# TODO: Implement the UserIntent schema for intent classification.
# This schema should include fields for intent_type, confidence, and reasoning.
# Refer to README.md Task 1.2 for detailed field requirements.
class UserIntent(BaseModel):
    """User intent classification - TO BE IMPLEMENTED"""
    pass


class ConversationTurn(BaseModel):
    """Represents a single turn in the conversation"""
    user_input: str
    agent_response: Any  # Can be any of the response types above
    intent: UserIntent
    tools_used: List[str] = Field(default_factory=lambda: list)
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionState(BaseModel):
    """Session state"""
    session_id: str
    user_id: str
    conversation_history: List[ConversationTurn] = Field(default_factory=lambda: list)
    document_context: List[str] = Field(default_factory=lambda: list, description="Active document IDs")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
