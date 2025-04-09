from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Any, Literal, Dict, get_type_hints
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from supabase import Client
import logfire
import os
import json

# Import the message classes from Pydantic AI
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter
)

from app_coder import app_coder, AppCoderDeps, list_documentation_pages_helper

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire='never')

# Configure API clients for both OpenAI and Deepseek
openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
deepseek_base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

# Initialize separate clients for each API
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=openai_base_url
)

# Deepseek client - we'll use the sync version for custom handling
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=deepseek_base_url
)

# Get model names from environment
reasoner_llm_model = os.getenv('REASONER_MODEL', 'deepseek-reasoner')
primary_llm_model = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')

# Set environment variables for OpenAI client
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = openai_base_url

# Create the agents with appropriate models
# For the reasoner, we'll create a custom function to use deepseek-reasoner directly
# rather than through the OpenAIModel abstraction, due to its special fields

router_agent = Agent(  
    OpenAIModel(primary_llm_model),
    system_prompt='Your job is to route the user message either to the end of the conversation or to continue coding the cross-platform application.',  
)

end_conversation_agent = Agent(  
    OpenAIModel(primary_llm_model),
    system_prompt='Your job is to end a conversation by providing a summary of the created app components, installation instructions, and a friendly goodbye.',  
)

platform_selection_agent = Agent(
    OpenAIModel(primary_llm_model),
    system_prompt='''Your job is to determine which platforms the user wants to target.
Options are: web (React), desktop (Electron), mobile (NativeScript), and server (Node.js).
Respond with a comma-separated list of platforms the user wants code for.'''
)

# Connect to Supabase
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Define state schema
class CodeperState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    scope: str
    platforms: List[str]

# Function to ensure all required state keys exist with defaults
def ensure_state_has_defaults(state):
    if not isinstance(state, dict):
        state = {}
    
    if 'latest_user_message' not in state:
        state['latest_user_message'] = ""
    
    if 'messages' not in state:
        state['messages'] = []
    
    if 'scope' not in state:
        state['scope'] = ""
    
    if 'platforms' not in state:
        state['platforms'] = []
    
    return state

# Custom function to call the deepseek-reasoner model directly
async def call_deepseek_reasoner(prompt):
    """
    Custom function to call the deepseek-reasoner model which has special output format.
    
    Args:
        prompt: The prompt text
        
    Returns:
        The final response content from the model
    """
    try:
        print(f"Calling deepseek-reasoner with prompt: {prompt[:100]}...")
        
        # Call deepseek-reasoner with special handling
        messages = [{"role": "user", "content": prompt}]
        
        # Use sync client since Pydantic-AI expects string output
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            max_tokens=4000  # Limit final response length (not reasoning chain)
        )
        
        # Get both reasoning chain and final content
        reasoning = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        
        print(f"Received reasoning ({len(reasoning)} chars) and content ({len(content)} chars)")
        
        # Save the reasoning to a file for inspection
        os.makedirs("workbench", exist_ok=True)
        with open(os.path.join("workbench", "reasoning_chain.md"), "w", encoding="utf-8") as f:
            f.write(f"# Reasoning Chain\n\n{reasoning}\n\n# Final Response\n\n{content}")
            
        # Return the final content for the scope
        return content
        
    except Exception as e:
        print(f"Error calling deepseek-reasoner: {str(e)}")
        # Fallback to a default response
        return f"Error generating scope: {str(e)}"

# Determine which platforms to target
async def select_platforms(state: CodeperState):
    # Ensure state has all required keys
    state = ensure_state_has_defaults(state)
    
    prompt = f"""
    The user is requesting an app with this description:
    
    {state['latest_user_message']}
    
    Determine which platforms should be targeted for this app. 
    Respond with just a comma-separated list of the platforms to target, selected from:
    - react (for web)
    - electron (for desktop)
    - nativescript (for mobile)
    - nodejs (for server)
    
    For example: "react,electron,nodejs"
    """
    
    result = await platform_selection_agent.run(prompt)
    platforms_str = result.data.strip()
    platforms = [p.strip().lower() for p in platforms_str.split(',')]
    
    # Validate platforms
    valid_platforms = ['react', 'electron', 'nativescript', 'nodejs']
    platforms = [p for p in platforms if p in valid_platforms]
    
    # Make sure we have at least one platform
    if not platforms:
        platforms = ['react']  # Default to React if no valid platform specified
        
    return {"platforms": platforms}

# Scope Definition Node with Deepseek Reasoner
async def define_scope_with_reasoner(state: CodeperState):
    # Ensure state has all required keys
    state = ensure_state_has_defaults(state)
    
    # First, get the documentation pages so the reasoner can decide which ones are necessary
    try:
        documentation_pages = await list_documentation_pages_helper(supabase)
        documentation_pages_str = "\n".join(documentation_pages)
    except Exception as e:
        print(f"Error retrieving documentation pages: {str(e)}")
        documentation_pages_str = "Documentation pages currently unavailable."

    platforms_str = ", ".join(state['platforms'])

    # Then, prepare the prompt for the reasoner
    prompt = f"""
    User App Request: {state['latest_user_message']}
    
    Target Platforms: {platforms_str}
    
    Create a detailed scope document for the cross-platform application including:
    - Architecture diagram
    - Core components for each platform ({platforms_str})
    - Shared components and logic
    - External dependencies and APIs
    - Data flow between components
    - User interface mockups
    
    Also, based on these documentation pages available:

    {documentation_pages_str}

    Include a list of documentation pages that are relevant to creating this app in the scope document,
    specifically focusing on the selected platforms: {platforms_str}.
    """

    # Call deepseek-reasoner directly with custom handling
    scope = await call_deepseek_reasoner(prompt)

    # Save the scope to a file
    scope_path = os.path.join("workbench", "scope.md")
    os.makedirs("workbench", exist_ok=True)

    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(scope)
    
    return {"scope": scope}

# Coding Node with Feedback Handling
async def coder_agent(state: CodeperState, writer):
    # Ensure state has all required keys
    state = ensure_state_has_defaults(state)
    
    # Prepare dependencies
    deps = AppCoderDeps(
        supabase=supabase,
        openai_client=openai_client,  # Use OpenAI client for coding
        reasoner_output=state['scope'],
        platforms=state['platforms']
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state.get('messages', []):
        try:
            message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))
        except Exception as e:
            print(f"Error processing message: {e}")
            continue

    # Run the agent in a stream
    async with app_coder.run_stream(
        state['latest_user_message'],
        deps=deps,
        message_history=message_history
    ) as result:
        # Stream partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            writer(chunk)

    return {"messages": [result.new_messages_json()]}

# Modified implementation to avoid the interrupt error
def get_next_user_message(state: CodeperState):
    """
    This is a placeholder for the interrupt node.
    In LangGraph, this is a special node that will pause execution
    and wait for the next user input.
    
    The actual interrupt happens in the streamlit_ui.py
    when it calls codeper_flow.astream with Command(resume=user_input)
    """
    # We don't call interrupt() here, as that's handled by the UI
    # Just return the state as-is
    return state

# Determine if the user is finished creating their app or not
async def route_user_message(state: CodeperState):
    # Ensure state has all required keys
    state = ensure_state_has_defaults(state)
    
    prompt = f"""
    The user has sent a message: 
    
    {state['latest_user_message']}

    If the user wants to end the conversation or indicates they're done with the application, respond with just the text "finish_conversation".
    If the user wants to continue coding the app, respond with just the text "coder_agent".
    """

    result = await router_agent.run(prompt)
    next_action = result.data.strip().lower()

    if next_action == "finish_conversation":
        return "finish_conversation"
    else:
        return "coder_agent"

# End of conversation agent to give instructions for executing the app
async def finish_conversation(state: CodeperState, writer):
    # Ensure state has all required keys
    state = ensure_state_has_defaults(state)
    
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state.get('messages', []):
        try:
            message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))
        except Exception as e:
            print(f"Error processing message: {e}")
            continue

    platforms_str = ", ".join(state['platforms'])
    
    # Custom prompt for the end conversation agent
    prompt = f"""
    The user has completed coding their application for the following platforms: {platforms_str}.
    
    Summarize what was created, provide instructions for running the application on each platform,
    and offer a friendly goodbye.
    
    Their original request was: {state['latest_user_message']}
    """

    # Run the agent in a stream
    async with end_conversation_agent.run_stream(
        prompt,
        message_history=message_history
    ) as result:
        # Stream partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            writer(chunk)

    return {"messages": [result.new_messages_json()]}        

# Build workflow
builder = StateGraph(CodeperState)

# Add nodes
builder.add_node("select_platforms", select_platforms)
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("coder_agent", coder_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("finish_conversation", finish_conversation)

# Set edges
builder.add_edge(START, "select_platforms")
builder.add_edge("select_platforms", "define_scope_with_reasoner")
builder.add_edge("define_scope_with_reasoner", "coder_agent")
builder.add_edge("coder_agent", "get_next_user_message")
builder.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    {"coder_agent": "coder_agent", "finish_conversation": "finish_conversation"}
)
builder.add_edge("finish_conversation", END)

# Configure persistence
memory = MemorySaver()
codeper_flow = builder.compile(checkpointer=memory)