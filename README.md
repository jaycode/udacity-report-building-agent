# DocDacity Intelligent Document Assistant

DocDacity is an interactive assistant for Q&A, summarization, and calculation tasks over simulated financial and healthcare documents. It uses LangChain, LangGraph, and OpenAI LLMs to orchestrate complex workflows and structured outputs, providing clear, source-cited answers and summaries.

---

## Implementation Decisions

- **LangGraph Workflow**: The assistant uses a stateful workflow graph, where user input is classified for intent (`qa`, `summarization`, `calculation`) and routed to specialized agent nodes.
- **Tool Use**: Agents use tools (`document_search`, `document_reader`, `calculator`, `document_statistics`) to retrieve information and perform calculations, with all usage logged for compliance.
- **Structured Output**: All agent responses are enforced to match Pydantic schemas (`AnswerResponse`, `SummarizationResponse`, `CalculationResponse`) for reliability and consistency.
- **Session & Memory**: Conversation history and active documents are persisted by session, enabling context-aware answers and memory summaries.

---

## How State & Memory Work

- **AgentState**: Tracks messages, intent, next step, conversation history, document context, responses, and tools used for each turn.
- **ConversationTurn**: Each user/assistant interaction is stored as a structured object, including intent, response, and tools used.
- **SessionState**: Manages overall session, history, current documents, and timestamps.
- **Memory Update**: After each turn, the agent updates conversation history, message history, and active documents, ensuring continuity across interactions.

# IMPORTANT NOTE

**Memory update** is already implemented in the `assistant` module

---

## Structured Output Enforcement

- Each agent node uses `llm.with_structured_output` with the relevant schema.
- Responses must include required fields (e.g., sources for QA, key points for summaries, step-by-step explanations for calculations).
- Invalid or missing fields are filled using workflow logic before finalizing the response.

---

## Example Conversations

### Q&A

**User:** What's the total amount in invoice INV-001?  
**Assistant:** The total amount in invoice INV-001 is $22,000.  
**Sources:** INV-001

### Summarization

**User:** Summarize all contracts.  
**Assistant:**  
DocDacity Solutions Inc. provides platform access, technical support, and monthly reports for Healthcare Partners LLC.  
**Key Points:**  
• 12-month contract  
• Monthly fee: $15,000  
• Total value: $180,000  
**Documents Analyzed:** CON-001

### Calculation

**User:** Calculate the sum of all invoice totals.  
**Assistant:**  
Step 1: Invoice INV-001 total is $22,000.  
Step 2: Invoice INV-002 total is $69,300.  
Step 3: Invoice INV-003 total is $214,500.  
Sum: $305,800

### Document Search

**User:** Find documents over $50,000  
**Assistant:** Found 2 documents: INV-002 ($69,300), INV-003 ($214,500)

---

## How to Run

1. Install requirements:  
   `pip install -r requirements.txt`
2. Ensure your OpenAI API key is in `.env`.
3. Start the CLI:  
   `python main.py`
4. Try commands like `/help`, `/history`, `/docs`, or ask questions as shown above.

---

## Features

- Q&A with source citation
- Document summarization
- Step-by-step financial calculations
- Session memory and context
- Document search and statistics
- Tool usage logging for compliance

---