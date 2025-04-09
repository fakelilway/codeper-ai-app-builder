from __future__ import annotations
from typing import Literal, TypedDict
from langgraph.types import Command
from openai import AsyncOpenAI
from supabase import Client
import streamlit as st
import logfire
import asyncio
import json
import uuid
import os
import random
from pathlib import Path
import traceback

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Codeper - Cross-Platform App Development",
    page_icon="🧪",
    layout="wide"
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Apply custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E4053;
    margin-bottom: 20px;
}
.sub-header {
    color: #5D6D7E;
    margin-bottom: 15px;
}
.file-browser {
    background-color: #F8F9F9;
    border-radius: 5px;
    padding: 15px;
    margin-top: 10px;
}
.example-box {
    background-color: #EBF5FB;
    border-radius: 5px;
    padding: 15px;
    margin-top: 20px;
}
.doc-links {
    margin-top: 30px;
}
.error-message {
    background-color: #FADBD8;
    color: #943126;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Initialize API clients
openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
deepseek_base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

# OpenAI client
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=openai_base_url
)

# Deepseek client
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=deepseek_base_url
)

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire='never')

# Supabase client setup with error handling
try:
    supabase: Client = Client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
except Exception as e:
    st.error(f"Failed to connect to Supabase: {str(e)}")
    supabase = None

# Now import the graph AFTER setting up environment and clients
try:
    from graph import codeper_flow
except Exception as e:
    st.error(f"Failed to load graph module: {str(e)}")
    traceback.print_exc()

@st.cache_resource
def get_thread_id():
    return str(uuid.uuid4())

thread_id = get_thread_id()

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # First message from user
    if len(st.session_state.messages) <= 1:
        # Create a full initial state
        initial_state = {
            "latest_user_message": user_input,
            "messages": [],  # Empty messages to start
            "scope": "",     # Empty scope to start
            "platforms": []  # Empty platforms to start
        }
        
        try:
            # Run the graph with the initial state
            async for msg in codeper_flow.astream(
                    initial_state, config, stream_mode="custom"
                ):
                    yield msg
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n\nPlease try again with a different request."
            yield error_msg
            print("Error trace:")
            traceback.print_exc()
    # Continue the conversation
    else:
        try:
            # Check if database is available for RAG
            table_exists = check_database_table()
            if not table_exists:
                yield "⚠️ Note: Documentation database tables (react_pages, electron_pages, etc.) are not available. Generation will continue without RAG support.\n\n"
            
            # Resume the conversation with the user's input
            async for msg in codeper_flow.astream(
                Command(resume={"latest_user_message": user_input}), config, stream_mode="custom"
            ):
                yield msg
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n\nPlease try again with a different request."
            yield error_msg
            print("Error trace:")
            traceback.print_exc()

def check_database_table():
    """Check if the documentation tables exist in Supabase."""
    if not supabase:
        return False
    
    try:
        # Try a simple query to check if any of the tables exist
        platform_tables = [
            "react_pages",
            "electron_pages",
            "node_pages",
            "native_script_pages"
        ]
        
        for table in platform_tables:
            try:
                result = supabase.table(table).select('count', count='exact').limit(1).execute()
                if result:
                    # At least one table exists, which is good enough
                    return True
            except Exception as e:
                print(f"Error checking {table}: {str(e)}")
                # Continue to check other tables
        
        # If we get here, none of the tables exist
        return False
    except Exception as e:
        print(f"Database table check failed: {str(e)}")
        return False

def get_file_info():
    """Get information about created files in the workbench directory."""
    workbench_dir = Path("workbench")
    if not workbench_dir.exists():
        return "No files created yet."
    
    platforms = ["react", "electron", "nodejs", "nativescript"]
    result = []
    
    # Check if scope.md exists
    scope_file = workbench_dir / "scope.md"
    if scope_file.exists():
        result.append(f"- 📄 [scope.md](file://{scope_file.absolute()})")
    
    # Check platform directories
    for platform in platforms:
        platform_dir = workbench_dir / platform
        if platform_dir.exists() and any(platform_dir.iterdir()):
            result.append(f"\n### {platform.capitalize()} Files:")
            for file in sorted(platform_dir.glob("**/*")):
                if file.is_file():
                    rel_path = file.relative_to(workbench_dir)
                    result.append(f"- 📄 [{rel_path}](file://{file.absolute()})")
    
    if not result:
        return "No files created yet."
    
    return "\n".join(result)

def get_example_requests():
    """Get example app requests for the user to try."""
    examples = [
        "Build a task management app with a React frontend and Node.js backend",
        "Create a markdown note-taking desktop application with Electron",
        "Develop a weather app for mobile devices using NativeScript",
        "Make a cross-platform chat application that works on web, desktop and mobile",
        "Build an e-commerce product catalog with filtering capabilities",
        "Create a PDF viewer and annotation tool for desktop with Electron",
        "Develop a recipe management app with ingredient search functionality"
    ]
    return random.sample(examples, 3)

def initialize_supabase_database():
    """Try to initialize the Supabase database if needed."""
    if not supabase:
        return "Supabase client not available"
    
    try:
        # Check if any tables exist
        if check_database_table():
            return "Database already initialized"
        
        # Here we could run the SQL to create the tables
        # But for security reasons, we'll just recommend manual setup
        return "Database needs initialization"
    except Exception as e:
        return f"Error checking database: {str(e)}"

async def main():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1 class='main-header'>Codeper - AI-Powered Cross-Platform App Development</h1>", unsafe_allow_html=True)
        st.markdown("<p>Describe the app you want to build, and I'll generate code for web, desktop, and mobile platforms.</p>", unsafe_allow_html=True)
        
        # Initialize chat history in session state if not present
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            message_type = message.get("type", "")
            if message_type in ["human", "ai", "user", "assistant"]:
                # Map message types to streamlit's expected format
                display_type = "user" if message_type in ["human", "user"] else "assistant"
                with st.chat_message(display_type):
                    st.markdown(message.get("content", ""))    

        # Chat input for the user
        user_input = st.chat_input("What app would you like to build today?")

        if user_input:
            # Add user message to session state
            st.session_state.messages.append({"type": "human", "content": user_input})
            
            # Display user prompt in the UI
            with st.chat_message("user"):
                st.markdown(user_input)

            # Display assistant response in chat message container
            response_content = ""
            with st.chat_message("assistant"):
                message_placeholder = st.empty()  # Placeholder for updating the message
                try:
                    # Run the async generator to fetch responses
                    async for chunk in run_agent_with_streaming(user_input):
                        if chunk:  # Only process non-empty chunks
                            response_content += chunk
                            # Update the placeholder with the current response content
                            message_placeholder.markdown(response_content)
                except Exception as e:
                    # Add an error message if something goes wrong
                    error_message = f"An error occurred: {str(e)}\n\nPlease try again with a different request."
                    response_content = error_message
                    message_placeholder.markdown(error_message)
                    traceback.print_exc()
            
            # Only add the response to the session state if it's not empty
            if response_content:
                st.session_state.messages.append({"type": "ai", "content": response_content})
    
    with col2:
        # Check database status
        db_status = initialize_supabase_database()
        if db_status == "Database needs initialization":
            st.markdown("<div class='error-message'>⚠️ Warning: The database needs to be initialized. Some features may not work correctly.</div>", unsafe_allow_html=True)
            st.button("Initialize Database", on_click=lambda: st.info("Please contact the administrator to set up the database"))
        
        st.markdown("<div class='file-browser'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Generated Files</h3>", unsafe_allow_html=True)
        file_info = get_file_info()
        st.markdown(file_info)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='example-box'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Try these examples</h3>", unsafe_allow_html=True)
        examples = get_example_requests()
        for example in examples:
            if st.button(f"📱 {example}", use_container_width=True):
                # Clear input and set this as the new query
                st.session_state.messages.append({"type": "human", "content": example})
                # Rerun to process the new query
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
                
        st.markdown("<div class='doc-links'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Platform Documentation</h3>", unsafe_allow_html=True)
        st.markdown("""
        - [React Documentation](https://react.dev/)
        - [Electron Documentation](https://www.electronjs.org/docs/latest/)
        - [Node.js Documentation](https://nodejs.org/en/learn)
        - [NativeScript Documentation](https://docs.nativescript.org/)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            # Also clear workbench directory if we want to
            # import shutil
            # shutil.rmtree("workbench", ignore_errors=True)
            st.rerun()

        # Add platform selection
        st.markdown("<div class='file-browser'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Project Info</h3>", unsafe_allow_html=True)
        
        # Get currently selected platforms 
        if "selected_platforms" not in st.session_state:
            st.session_state.selected_platforms = []
            
        try:
            # Check if we can get info from scope.md
            scope_file = Path("workbench/scope.md")
            if scope_file.exists():
                with open(scope_file, "r") as f:
                    scope_content = f.read()
                    # Extract platforms using simple text search
                    if "Target Platforms:" in scope_content:
                        platform_line = scope_content.split("Target Platforms:")[1].split("\n")[0].strip()
                        st.markdown(f"**Target Platforms:** {platform_line}")
        except Exception as e:
            st.write("No project scope found yet.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show API information
        st.markdown("<div class='file-browser'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>API Configuration</h3>", unsafe_allow_html=True)
        
        # Display the current API configurations
        st.markdown("**OpenAI API:**")
        st.markdown(f"- Base URL: {openai_base_url}")
        st.markdown(f"- Model: {os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')}")
        
        st.markdown("**Deepseek API:**")
        st.markdown(f"- Base URL: {deepseek_base_url}")
        st.markdown(f"- Model: {os.getenv('REASONER_MODEL', 'deepseek-llm-67b-chat')}")
        
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    asyncio.run(main())