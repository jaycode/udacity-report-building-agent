import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langchain_openai import ChatOpenAI

from schemas import SessionState, ConversationTurn
from retrieval import SimulatedRetriever
from tools import get_all_tools, ToolLogger
from agent import create_workflow, AgentState
from prompts import MEMORY_SUMMARY_PROMPT


class DocumentAssistant:
    """
    The main assistant.
    """
    
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        session_storage_path: str = "./sessions"
    ):
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
            base_url="https://openai.vocareum.com/v1"
        )
        
        # Initialize components
        self.retriever = SimulatedRetriever()
        self.tool_logger = ToolLogger(logs_dir="./logs")
        self.tools = get_all_tools(self.retriever, self.tool_logger)
        
        # Create workflow
        self.workflow = create_workflow(self.llm, self.tools)
        
        # Session management
        self.session_storage_path = session_storage_path
        os.makedirs(session_storage_path, exist_ok=True)
        
        # Current session
        self.current_session: Optional[SessionState] = None
        
    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Start a new session or resume an existing one.
        """
        if session_id and self._session_exists(session_id):
            # Load existing session
            self.current_session = self._load_session(session_id)
            print(f"Resumed session {session_id}")
        else:
            # Create new session
            session_id = session_id or str(uuid.uuid4())
            self.current_session = SessionState(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                document_context=[]
            )
            print(f"Started new session {session_id}")


        # Create a new ToolLogger with session-specific logging
        self.tool_logger = ToolLogger(logs_dir="./logs", session_id=session_id)
        self.tools = get_all_tools(self.retriever, self.tool_logger)

        # Recreate workflow with new tools
        self.workflow = create_workflow(self.llm, self.tools)
        
        return session_id
    
    def _session_exists(self, session_id: str) -> bool:
        """Check if a session file exists"""
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        return os.path.exists(filepath)
    
    def _load_session(self, session_id: str) -> SessionState:
        """Load session from file"""
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        with open(filepath, 'r') as f:
            data = json.load(f)
        return SessionState(**data)
    
    def _save_session(self):
        """Save current session to file"""
        if self.current_session:
            filepath = os.path.join(
                self.session_storage_path,
                f"{self.current_session.session_id}.json"
            )
            # Convert to dict and handle datetime serialization
            session_dict = self.current_session.dict()
            
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(filepath, 'w') as f:
                json.dump(session_dict, f, indent=2, default=serialize_datetime)
    
    def _get_conversation_summary(self) -> str:
        """
        Generate a summary of the conversation.
        """
        if not self.current_session or not self.current_session.conversation_history:
            return "No previous conversation."
        
        # Format recent conversation
        history_text = ""
        for turn in self.current_session.conversation_history[-5:]:  # Last 5 turns
            history_text += f"User: {turn.user_input}\n"
            
            # Extract response text based on type
            if isinstance(turn.agent_response, dict):
                if "answer" in turn.agent_response:
                    history_text += f"Assistant: {turn.agent_response['answer']}\n"
                elif "summary" in turn.agent_response:
                    history_text += f"Assistant: {turn.agent_response['summary']}\n"
                elif "explanation" in turn.agent_response:
                    history_text += f"Assistant: {turn.agent_response['explanation']}\n"
        
        # Use LLM to summarize if history is long
        if len(self.current_session.conversation_history) > 3:
            prompt = MEMORY_SUMMARY_PROMPT.format(
                conversation_history=history_text,
                max_length=100
            )
            summary = self.llm.invoke(prompt).content
            return summary
        
        return history_text
    
    def process_message(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user message.
        """
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")
        
        # Prepare initial state
        initial_state: AgentState = {
            "messages": [],
            "user_input": user_input,
            "intent": None,
            "next_step": "classify_intent",
            "conversation_history": self.current_session.conversation_history,
            "conversation_summary": self._get_conversation_summary(),
            "active_documents": self.current_session.document_context,
            "current_response": None,
            "tools_used": [],
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id
        }
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Update session with new turn
            if final_state.get("current_response"):
                turn = ConversationTurn(
                    user_input=user_input,
                    agent_response=final_state["current_response"],
                    intent=final_state.get("intent"),
                    tools_used=final_state.get("tools_used", [])
                )
                self.current_session.conversation_history.append(turn)
                self.current_session.last_updated = datetime.now()
                
                # Update document context
                if final_state.get("active_documents"):
                    self.current_session.document_context = list(set(
                        self.current_session.document_context + 
                        final_state["active_documents"]
                    ))
                
                # Save session
                self._save_session()
            
            # Return response
            return {
                "success": True,
                "response": final_state.get("current_response"),
                "intent": final_state.get("intent").dict() if final_state.get("intent") else None,
                "tools_used": final_state.get("tools_used", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def get_session_history(self) -> List[ConversationTurn]:
        """Get the conversation history for the current session"""
        if self.current_session:
            return self.current_session.conversation_history
        return []
    
    def export_logs(self, filepath: str):
        """Export tool usage logs for compliance"""
        self.tool_logger.save_logs(filepath)
    
    def add_document(self, doc_id: str, title: str, content: str, doc_type: str, metadata: Dict[str, Any] = None):
        """
        Add a new document to the retriever.
        """
        from src.retrieval import Document
        
        doc = Document(
            doc_id=doc_id,
            title=title,
            content=content,
            doc_type=doc_type,
            metadata=metadata or {}
        )
        self.retriever.add_document(doc)
        print(f"Added document {doc_id} to the system")


# Example usage function
def create_assistant(api_key: str) -> DocumentAssistant:
    """Create a configured assistant"""
    return DocumentAssistant(
        openai_api_key=api_key,
        model_name="gpt-4o",
        temperature=0.1,
        session_storage_path="./sessions"
    )
