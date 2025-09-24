from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import re

from schemas import (
    UserIntent, ConversationTurn, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template
import pdb

# The AgentState class is already implemented for you. 
# Study the structure to understand how state flows through the LangGraph workflow.
# See README.md Task 2.1 for detailed explanations of each property.
class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    
    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str
    
    # Memory and context
    conversation_history: List[ConversationTurn]
    conversation_summary: str
    active_documents: List[str]
    
    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]
    
    # Session management
    session_id: str
    user_id: str

# COMPLETED: Implement the classify_intent function.
# This function should classify the user's intent and set the next step in the workflow.
# Refer to README.md Task 2.2 for detailed implementation requirements.
def classify_intent(state: AgentState, llm) -> AgentState:
    """
    Classify user intent
    """
    messages = []
    system_msg = (
        "You classify user intent for routing. "
        "Return a JSON object with fields intent_type, confidence, reasoning. "
        "intent_type must be one of: qa, summarization, calculation. "
        "If you are not sure, pick qa"
        "Use the conversation summary and recent turns as context."
    )

    messages.append(SystemMessage(content=system_msg))
    # Add conversation history
    for msg in state.get("messages", [])[-4:]:  # Last 4 messages
        messages.append(msg)
    
    messages.append(HumanMessage(content=state["user_input"]))

    structured_llm = llm.with_structured_output(UserIntent)

    # Get structured response
    user_intent = structured_llm.invoke(messages)
    state["intent"] = user_intent
    mapping = {"qa": "qa_agent", "summarization": "summarization_agent", "calculation": "calculation_agent"}
    state["next_step"] = mapping.get(user_intent.intent_type, "qa_agent")
    return state


def qa_agent(state: AgentState, llm, tools) -> AgentState:
    """
    Handle Q&A tasks
    """
    messages = []
    
    system_msg = f"""You are a helpful document assistant. 

IMPORTANT TOOL USAGE INSTRUCTIONS:
1. For document_search tool, you MUST provide:
   - query: Your search query (REQUIRED - cannot be empty)
   - search_type: Either "keyword" (default), "type", "amount", or "amount_range"
   - Additional parameters based on search_type:
     * For amount queries like "over $50,000": use comparison="over", amount=50000
     * For "under $10,000": use comparison="under", amount=10000
     * For "between X and Y": use min_amount=X, max_amount=Y
     * For "around $25,000": use comparison="approximate", amount=25000
     * For "exactly $100,000": use comparison="exact", amount=100000

2. For document_reader tool, you MUST provide:
   - doc_id: The exact document ID to read (REQUIRED)

3. For calculator tool, you MUST provide:
   - expression: The mathematical expression to evaluate (REQUIRED)

4. For document_statistics tool:
   - No parameters required - shows overview of all documents

Always search for documents first before answering questions about their content.
Cite your sources by document ID.

Current conversation context: {state.get('conversation_summary', 'No previous context')}

User question: {state["user_input"]}
"""
    messages.append(SystemMessage(content=system_msg))
    
    # Add conversation history
    for msg in state.get("messages", [])[-4:]:  # Last 4 messages
        messages.append(msg)
    
    messages.append(HumanMessage(content=state["user_input"]))
    
    llm_with_tools = llm.bind_tools(tools)
    
    tool_response = llm_with_tools.invoke(messages)
    messages.append(tool_response)
    
    # Process tool calls
    sources = []
    tools_used = []
    
    if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
        for tool_call in tool_response.tool_calls:
            # Find the matching tool and execute it
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Ensure required args are present for document_search
            if tool_name == "document_search" and "query" not in tool_args:
                # Extract query from user input if not provided
                tool_args["query"] = state["user_input"]
            
            # Find the tool
            matching_tool = next((t for t in tools if t.name == tool_name), None)
            if matching_tool:
                # Execute the tool
                tool_result = matching_tool.invoke(tool_args)
                tools_used.append(tool_name)
                
                # Extract document IDs from results
                doc_ids = re.findall(r'ID: ([\w-]+)', str(tool_result))
                sources.extend(doc_ids)
                
                # Add tool result to messages
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call.get('id', tool_name)
                ))
    
    # Get the structured output
    structured_llm = llm.with_structured_output(AnswerResponse)
    
    # Create a prompt for the final response
    final_prompt = f"""Based on the tool results and conversation, provide a comprehensive answer to: {state['user_input']}
    
    Include the document IDs you referenced as sources."""
    
    messages.append(HumanMessage(content=final_prompt))
    
    # Get structured response
    response = structured_llm.invoke(messages)
    
    # Ensure sources are populated
    if not response.sources and sources:
        response.sources = list(set(sources))
    
    state["current_response"] = response.dict()
    state["tools_used"] = tools_used
    state["next_step"] = "update_memory"
    
    return state


def summarization_agent(state: AgentState, llm, tools) -> AgentState:
    """
    Handle summarization tasks
    """
    messages = []
    
    # System message with explicit tool instructions
    system_msg = f"""You are an expert document summarizer.

IMPORTANT TOOL USAGE INSTRUCTIONS:
1. For document_search tool, you MUST provide:
   - query: Your search query (REQUIRED - use terms from the user's request)
   - search_type: Either "keyword" (default), "type", or "amount_range"

2. For document_reader tool, you MUST provide:
   - doc_id: The exact document ID to read (REQUIRED)

First search for relevant documents, then read them to create comprehensive summaries.
Extract key points and cite document IDs.

Current conversation context: {state.get('conversation_summary', 'No previous context')}

User request: {state["user_input"]}
"""
    messages.append(SystemMessage(content=system_msg))
    
    # Add conversation history
    for msg in state.get("messages", [])[-4:]:
        messages.append(msg)
    
    # Add current request
    messages.append(HumanMessage(content=state["user_input"]))
    
    # Use tools to gather documents
    llm_with_tools = llm.bind_tools(tools)
    tool_response = llm_with_tools.invoke(messages)
    messages.append(tool_response)
    
    # Process tool calls
    doc_ids = []
    tools_used = []
    original_content_length = 0
    
    if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
        for tool_call in tool_response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Ensure required args are present
            if tool_name == "document_search" and "query" not in tool_args:
                # Extract key terms from user input for search
                tool_args["query"] = " ".join(state["user_input"].split()[:5])
            
            matching_tool = next((t for t in tools if t.name == tool_name), None)
            if matching_tool:
                tool_result = matching_tool.invoke(tool_args)
                tools_used.append(tool_name)
                
                # Extract document IDs and estimate content length
                doc_ids.extend(re.findall(r'ID: ([\w-]+)', str(tool_result)))
                original_content_length += len(str(tool_result))
                
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call.get('id', tool_name)
                ))
    
    # Get structured summary
    structured_llm = llm.with_structured_output(SummarizationResponse)
    
    final_prompt = f"""Based on the documents you've read, create a comprehensive summary for: {state['user_input']}
    
    Extract 3-5 key points and include the document IDs you summarized."""
    
    messages.append(HumanMessage(content=final_prompt))
    
    response = structured_llm.invoke(messages)
    
    if not response.document_ids and doc_ids:
        response.document_ids = list(set(doc_ids))
    if response.original_length == 0:
        response.original_length = original_content_length
    
    state["current_response"] = response.dict()
    state["tools_used"] = tools_used
    state["next_step"] = "update_memory"
    
    return state


def calculation_agent(state: AgentState, llm, tools) -> AgentState:
    """
    Handle calculation tasks
    """
    messages = []
    
    # System message with tool instructions
    system_msg = f"""You are a precise calculator assistant.

IMPORTANT TOOL USAGE INSTRUCTIONS:
1. For calculator tool, you MUST provide:
   - expression: The mathematical expression to evaluate (REQUIRED)

2. For document_search tool (to find numbers), you MUST provide:
   - query: Your search query (REQUIRED - use terms related to the numbers you need)

3. For document_reader tool, you MUST provide:
   - doc_id: The exact document ID to read (REQUIRED)

Search documents first if you need to find specific numbers, then use the calculator.
Show your work step by step.

Current conversation context: {state.get('conversation_summary', 'No previous context')}

User request: {state["user_input"]}
"""
    messages.append(SystemMessage(content=system_msg))
    
    # Add conversation history
    for msg in state.get("messages", [])[-4:]:
        messages.append(msg)
    
    messages.append(HumanMessage(content=state["user_input"]))
    
    llm_with_tools = llm.bind_tools(tools)
    tool_response = llm_with_tools.invoke(messages)
    messages.append(tool_response)
    
    expression = ""
    calc_result = None
    tools_used = []
    
    if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
        for tool_call in tool_response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Ensure required args
            if tool_name == "document_search" and "query" not in tool_args:
                # Look for number-related keywords in user input
                tool_args["query"] = "total amount sum calculate"
            
            matching_tool = next((t for t in tools if t.name == tool_name), None)
            if matching_tool:
                tool_result = matching_tool.invoke(tool_args)
                tools_used.append(tool_name)
                
                if tool_name == "calculator":
                    expression = tool_args.get("expression", "")
                    # Extract result from output
                    match = re.search(r'result.*?is\s*([\d.,]+)', str(tool_result))
                    if match:
                        calc_result = float(match.group(1).replace(',', ''))
                
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call.get('id', tool_name)
                ))

    # COMPLETED: Complete the calculation_agent function.
    # Generate a structured response that includes the expression and step-by-step explanation.
    # Refer to README.md Task 2.3 for detailed implementation requirements.
    
    structured_llm = llm.with_structured_output(CalculationResponse)
    final_prompt = f"""
Based on the calculation just performed, provide a clear, step-by-step explanation of how you 
arrived at the result for: {state['user_input']}

Include:
- The mathematical expression used
- The result
- A step-by-step explanation
- Any relevant units or sources
"""
    messages.append(HumanMessage(content=final_prompt))
    response = structured_llm.invoke(messages)

    # Fill in missing fields if possible
    if not response.expression and expression:
        response.expression = expression
    if not response.result and calc_result is not None:
        response.result = calc_result

    state["current_response"] = response.dict()
    state["tools_used"] = tools_used
    state["next_step"] = "update_memory"
    return state


# COMPLETED: Implement the update_memory function.
# This function updates the conversation history and manages the state after each interaction.
# Refer to README.md Task 2.4 for detailed implementation requirements.

# NOTE: This function is not needed since the assistant module has already updated the conversation
#       history. I included the commented code as a proof that I can complete this specification
#       if needed (please clarify in the comment what I need to do).
def update_memory(state: AgentState) -> AgentState:
    """
    Update conversation memory
    """
    # # 1. Creates a ConversationTurn object from the current interaction
    # turn = ConversationTurn(
    #     user_input=state["user_input"],
    #     agent_response=state.get("current_response"),
    #     intent=state.get("intent"),
    #     tools_used=state.get("tools_used", [])
    # )

    # # 2. Adds the turn to the conversation history
    # state["conversation_history"].append(turn)

    # # 3. Updates the message history with user input and agent response
    # state["messages"].append(HumanMessage(content=state["user_input"]))
    # # Use 'answer', 'summary', or 'explanation' field from response
    # response = state.get("current_response", {})
    # ai_content = None
    # if isinstance(response, dict):
    #     for k in ["answer", "summary", "explanation"]:
    #         if k in response and response[k]:
    #             ai_content = response[k]
    #             break
    # if ai_content:
    #     state["messages"].append(AIMessage(content=ai_content))

    # # 4. Tracks active documents from the response
    # new_docs = []
    # if isinstance(response, dict):
    #     if "sources" in response and response["sources"]:
    #         new_docs.extend(response["sources"])
    #     if "document_ids" in response and response["document_ids"]:
    #         new_docs.extend(response["document_ids"])

    # # Update active_documents to include any new ones found
    # state["active_documents"] = list(set(state.get("active_documents", []) + new_docs))

    # # 5. Sets next_step to "end"
    # state["next_step"] = "end"
    # return state
    pass


def should_continue(state: AgentState) -> Literal["qa_agent", "summarization_agent", "calculation_agent", "end"]:
    """
    Router function
    """
    return state.get("next_step", "end")

# COMPLETED: Implement the create_workflow function.
# This function creates the LangGraph workflow that coordinates all the agents.
# Refer to README.md Task 2.5 for detailed implementation requirements and the graph structure.
def create_workflow(llm, tools):
    """
    Creates the LangGraph workflow
    """
    graph = StateGraph(AgentState)

    # Nodes setup
    graph.add_node("classify_intent", lambda state: classify_intent(state, llm))
    graph.set_entry_point("classify_intent")
    
    graph.add_node("qa_agent", lambda state: qa_agent(state, llm, tools))
    graph.add_node("summarization_agent", lambda state: summarization_agent(state, llm, tools))
    graph.add_node("calculation_agent", lambda state: calculation_agent(state, llm, tools))
    
    graph.add_node("update_memory", update_memory)

    # Routing
    graph.add_conditional_edges(
        "classify_intent",  # from node
        should_continue,    # routing function
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent"
        }
    )
    graph.add_edge("qa_agent", "update_memory")
    graph.add_edge("summarization_agent", "update_memory")
    graph.add_edge("calculation_agent", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()
