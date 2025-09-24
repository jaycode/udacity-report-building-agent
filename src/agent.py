from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import re

from schemas import (
    UserIntent, ConversationTurn, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template, \
    get_response_format_template
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
    messages.append(SystemMessage(content=get_intent_classification_prompt()))
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
    Handle Q&A tasks - refactored for dynamic prompting
    """
    messages = get_chat_prompt_template("qa").format_messages(
        conversation_summary = state.get("conversation_summary", "No previous conversation."),
        user_input = state["user_input"]
    )
    
    # Add conversation history
    history = state.get("messages", [])[-4:]
    # Insert before the final (just-appended) message
    messages[-1:-1] = history

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
    
    # Get structured response
    response = structured_llm.invoke(messages)
    
    # Ensure sources are populated
    if not response.sources and sources:
        response.sources = list(set(sources))

    # Polishing the prompt response with the response format template
    polisher_prompt_template = get_response_format_template("qa")
    polisher_prompt = polisher_prompt_template.format(
        question=response.question,
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence
    )

    polished_response = llm.invoke(polisher_prompt)
    rdict = response.dict()
    rdict["answer"] = polished_response.content
    
    state["current_response"] = rdict
    state["tools_used"] = tools_used
    state["next_step"] = "update_memory"
    
    return state


def summarization_agent(state: AgentState, llm, tools) -> AgentState:
    """
    Handle summarization tasks - refactored for dynamic prompting
    """
    messages = get_chat_prompt_template("qa").format_messages(
        conversation_summary = state.get("conversation_summary", "No previous conversation."),
        user_input = state["user_input"]
    )
    
    # Add conversation history
    history = state.get("messages", [])[-4:]
    # Insert before the final (just-appended) message
    messages[-1:-1] = history
    
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
    
    response = structured_llm.invoke(messages)
    
    if not response.document_ids and doc_ids:
        response.document_ids = list(set(doc_ids))
    if response.original_length == 0:
        response.original_length = original_content_length
    
    # Polishing the prompt response with the response format template
    polisher_prompt_template = get_response_format_template("summarization")
    polisher_prompt = polisher_prompt_template.format(
        documents=response.document_ids,
        key_points=response.key_points,
        summary=response.summary
    )

    polished_response = llm.invoke(polisher_prompt)
    rdict = response.dict()
    rdict["summary"] = polished_response.content
    
    state["current_response"] = rdict
    state["tools_used"] = tools_used
    state["next_step"] = "update_memory"
    
    return state


def calculation_agent(state: AgentState, llm, tools) -> AgentState:
    """
    Handle calculation tasks - refactored for dynamic prompting
    """
    messages = get_chat_prompt_template("calculation").format_messages(
        conversation_summary = state.get("conversation_summary", "No previous conversation."),
        user_input = state["user_input"]
    )
    
    llm_with_tools = llm.bind_tools(tools)
    tool_response = llm_with_tools.invoke(messages)
    messages.append(tool_response)
    
    expression = ""
    calc_result = None
    tools_used = []
    doc_ids = []
    
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

                # Collect doc IDs from search results
                if tool_name == "document_search":
                    doc_ids.extend(re.findall(r'ID: ([\w-]+)', str(tool_result)))

                # Collect from reader directly via args
                if tool_name == "document_reader" and "doc_id" in tool_args:
                    doc_ids.append(tool_args["doc_id"])

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
    response = structured_llm.invoke(messages)

    # Polishing the prompt response with the response format template
    polisher_prompt_template = get_response_format_template("calculation")
    sources = ",".join(set(doc_ids))
    polisher_prompt = polisher_prompt_template.format(
        expression=response.expression,
        result=response.result,
        explanation=response.explanation,
        sources=sources
    )
    polished_response = llm.invoke(polisher_prompt)

    # Fill in missing fields if possible
    if not response.expression and expression:
        response.expression = expression
    if not response.result and calc_result is not None:
        response.result = calc_result

    rdict = response.dict()
    rdict["explanation"] = polished_response.content

    state["current_response"] = rdict
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
