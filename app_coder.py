from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import json
from pathlib import Path

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Dict, Any

load_dotenv()

llm = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY', 'no-llm-api-key-provided')
model = OpenAIModel(llm)
embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

logfire.configure(send_to_logfire='if-token-present')

is_ollama = "localhost" in base_url.lower()

@dataclass
class AppCoderDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    reasoner_output: str
    platforms: List[str]

system_prompt = """
[ROLE AND CONTEXT]
You are a specialized cross-platform application developer. Your expertise spans multiple platforms:
- Web apps (React.js)
- Desktop apps (Electron)
- Mobile apps (NativeScript)
- Server-side apps (Node.js)

You generate complete, production-ready code for applications across these platforms.

[CORE RESPONSIBILITIES]
1. App Development
   - Create cross-platform applications based on user requirements
   - Generate code for specific platforms as requested
   - Ensure consistency across platforms where appropriate
   - Optimize code for each platform's capabilities

2. Documentation Integration
   - Use RAG to search platform-specific documentation
   - Follow best practices for each platform
   - Integrate appropriate libraries and frameworks
   - Validate implementations against current standards

[CODE STRUCTURE AND DELIVERABLES]
For each platform, provide complete implementation files:

1. Web (React):
   - App.jsx/tsx: Main application component
   - Component files: UI components
   - API integration: Data fetching and state management
   - package.json: Dependencies

2. Desktop (Electron):
   - main.js: Main process
   - preload.js: Preload script
   - renderer files: UI implementation
   - package.json: Dependencies

3. Mobile (NativeScript):
   - app.js: Main application entry
   - views: UI components
   - Services: Data and business logic
   - package.json: Dependencies

4. Server (Node.js):
   - server.js: Main server implementation
   - routes/: API endpoints
   - models/: Data models
   - controllers/: Business logic
   - package.json: Dependencies

[DOCUMENTATION WORKFLOW]
1. Initial Research
   - Use RAG search for relevant platform documentation
   - Retrieve specific documentation pages for necessary components
   - Cross-reference examples for best practices
   - Prioritize official platform documentation

2. Implementation
   - Generate complete, working code without placeholders
   - Implement error handling and validation
   - Include comments explaining key concepts
   - Ensure cross-platform compatibility where required

3. Quality Assurance
   - Verify platform-specific implementations
   - Ensure proper error handling
   - Validate environment setup instructions
   - Check for security best practices

[INTERACTION GUIDELINES]
- Take initiative to implement complete solutions
- Be honest about platform limitations
- Provide clear explanations for technical decisions
- Suggest improvements beyond the initial requirements
- Explain cross-platform considerations

[ERROR HANDLING]
- Implement appropriate error handling for each platform
- Provide clear error messages
- Include recovery strategies
- Ensure consistent error management across platforms

[BEST PRACTICES]
- Follow platform-specific naming conventions
- Implement proper type safety
- Include comprehensive documentation
- Follow clean code principles
- Focus on maintainability and scalability
"""

app_coder = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=AppCoderDeps,
    retries=2
)

@app_coder.system_prompt  
def add_reasoner_output(ctx: RunContext[AppCoderDeps]) -> str:
    platforms_str = ", ".join(ctx.deps.platforms)
    return f"""
    \n\nAdditional app scope information from the reasoning agent. 
    This includes architecture, components, and data flow for the following platforms: {platforms_str}
    
    {ctx.deps.reasoner_output}
    """

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@app_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[AppCoderDeps], user_query: str, platform: str = None) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        platform: Optional filter for platform-specific docs (react, electron, nodejs, nativescript)
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Map platform to the correct table name
        table_name = None
        if platform:
            platform_table_map = {
                "react": "react_pages",
                "electron": "electron_pages",
                "nodejs": "node_pages",
                "nativescript": "native_script_pages"
            }
            
            table_name = platform_table_map.get(platform.lower())
        
        # Get results from appropriate tables
        formatted_chunks = []
        
        if table_name:
            # Query specific platform table
            try:
                # Check if the custom similarity function exists
                has_match_function = True
                try:
                    # Try calling the similarity function first
                    result = ctx.deps.supabase.rpc(
                        f'match_{table_name}',
                        {
                            'query_embedding': query_embedding,
                            'match_count': 5
                        }
                    ).execute()
                except Exception:
                    has_match_function = False
                
                if not has_match_function:
                    # Fall back to direct table query
                    result = ctx.deps.supabase.from_(table_name).select('*').limit(5).execute()
                
                if result.data:
                    platform_name = platform.capitalize()
                    if platform.lower() == 'nodejs':
                        platform_name = 'Node.js'
                    elif platform.lower() == 'nativescript':
                        platform_name = 'NativeScript'
                    
                    for doc in result.data:
                        chunk_text = f"""
# {doc.get('title', 'Documentation')} ({platform_name})

Source: {doc.get('url', 'Unknown URL')}

{doc.get('content', 'No content available')}
"""
                        formatted_chunks.append(chunk_text)
            except Exception as e:
                print(f"Error querying {table_name}: {str(e)}")
        else:
            # Query all platform tables
            for p, table in {
                "React": "react_pages",
                "Electron": "electron_pages",
                "Node.js": "node_pages",
                "NativeScript": "native_script_pages"
            }.items():
                try:
                    result = ctx.deps.supabase.from_(table).select('*').limit(2).execute()
                    
                    for doc in result.data:
                        chunk_text = f"""
# {doc.get('title', 'Documentation')} ({p})

Source: {doc.get('url', 'Unknown URL')}

{doc.get('content', 'No content available')}
"""
                        formatted_chunks.append(chunk_text)
                except Exception as e:
                    print(f"Error querying {table}: {str(e)}")
                    continue
            
        if not formatted_chunks:
            return f"No relevant documentation found for {platform if platform else 'any platform'}."
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


async def list_documentation_pages_helper(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available documentation pages across all platforms.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query each platform-specific table instead of site_pages
        platform_tables = [
            "react_pages",
            "electron_pages",
            "node_pages",
            "native_script_pages"
        ]
        
        # Collect URLs from all tables
        all_urls = []
        
        for table in platform_tables:
            try:
                result = supabase.from_(table).select('url, metadata').execute()
                
                if result.data:
                    for doc in result.data:
                        platform = table.replace('_pages', '').capitalize()
                        if platform == 'Native_script':
                            platform = 'NativeScript'
                        if platform == 'Node':
                            platform = 'Node.js'
                            
                        all_urls.append(f"{platform}: {doc['url']}")
            except Exception as e:
                print(f"Error querying {table}: {str(e)}")
                continue
        
        # Return unique URLs
        return sorted(set(all_urls))
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {str(e)}")
        return []

@app_coder.tool
async def list_documentation_pages(ctx: RunContext[AppCoderDeps], platform: str = None) -> List[str]:
    """
    Retrieve a list of available documentation pages, optionally filtered by platform.
    
    Args:
        ctx: The context including the Supabase client
        platform: Optional platform filter (react, electron, nodejs, nativescript)
        
    Returns:
        List[str]: List of documentation pages
    """
    try:
        # Map platform input to table name
        table_name = None
        if platform:
            platform_table_map = {
                "react": "react_pages",
                "electron": "electron_pages",
                "nodejs": "node_pages",
                "nativescript": "native_script_pages"
            }
            table_name = platform_table_map.get(platform.lower())
        
        # Collect URLs for the specified platform or all platforms
        all_urls = []
        
        if table_name:
            # Query specific platform
            result = ctx.deps.supabase.from_(table_name).select('url').execute()
            
            if result.data:
                platform_name = platform.capitalize()
                if platform.lower() == 'nodejs':
                    platform_name = 'Node.js'
                elif platform.lower() == 'nativescript':
                    platform_name = 'NativeScript'
                
                for doc in result.data:
                    all_urls.append(f"{platform_name}: {doc['url']}")
        else:
            # Query all platforms
            platform_tables = [
                ("React", "react_pages"),
                ("Electron", "electron_pages"),
                ("Node.js", "node_pages"),
                ("NativeScript", "native_script_pages")
            ]
            
            for platform_name, table in platform_tables:
                try:
                    result = ctx.deps.supabase.from_(table).select('url').execute()
                    
                    if result.data:
                        for doc in result.data:
                            all_urls.append(f"{platform_name}: {doc['url']}")
                except Exception:
                    # Skip tables that don't exist
                    continue
        
        return sorted(set(all_urls))
        
    except Exception as e:
        print(f"Error listing documentation pages: {e}")
        return []

@app_coder.tool
async def get_page_content(ctx: RunContext[AppCoderDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Try to find the page in each platform table
        platform_tables = [
            ("React", "react_pages"),
            ("Electron", "electron_pages"),
            ("Node.js", "node_pages"),
            ("NativeScript", "native_script_pages")
        ]
        
        for platform_name, table in platform_tables:
            try:
                # Query for chunks from this URL in this table
                result = ctx.deps.supabase.from_(table) \
                    .select('title, content, chunk_number') \
                    .eq('url', url) \
                    .order('chunk_number') \
                    .execute()
                
                if result.data:
                    # Found the content in this table
                    page_title = result.data[0]['title']
                    if " - " in page_title:
                        page_title = page_title.split(' - ')[0]  # Get the main title
                        
                    formatted_content = [f"# {page_title} ({platform_name})\n\nSource: {url}\n"]
                    
                    # Add each chunk's content
                    for chunk in result.data:
                        formatted_content.append(chunk['content'])
                        
                    # Join everything together
                    return "\n\n".join(formatted_content)
            except Exception:
                # Skip tables that don't exist
                continue
        
        return f"No content found for URL: {url}"
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
    
@app_coder.tool
async def save_code_to_file(ctx: RunContext[AppCoderDeps], filename: str, content: str, platform: str) -> str:
    """
    Save generated code to a file in a platform-specific directory.
    
    Args:
        ctx: The run context
        filename: Name of the file to save (including extension)
        content: Code content to save
        platform: Target platform (react, electron, nodejs, nativescript)
        
    Returns:
        str: Status message
    """
    try:
        # Create platform-specific directory if it doesn't exist
        base_dir = Path("workbench") / platform.lower()
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle nested directories in filename
        file_path = base_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Successfully saved {filename} for {platform} platform at {file_path}"
    
    except Exception as e:
        return f"Error saving file: {str(e)}"

@app_coder.tool
async def list_platform_files(ctx: RunContext[AppCoderDeps], platform: str = None) -> str:
    """
    List files that have been created for a specific platform or all platforms.
    
    Args:
        ctx: The run context
        platform: Optional platform to filter (react, electron, nodejs, nativescript)
        
    Returns:
        str: Formatted list of files
    """
    try:
        # Define base directory
        base_dir = Path("workbench")
        
        # Get platforms to check
        platforms_to_check = []
        if platform:
            platforms_to_check = [platform.lower()]
        else:
            # Check all valid platforms
            valid_platforms = ["react", "electron", "nodejs", "nativescript"]
            platforms_to_check = [p for p in valid_platforms if (base_dir / p).exists()]
            
        # Build result
        result = []
        for p in platforms_to_check:
            platform_dir = base_dir / p
            if platform_dir.exists():
                files = list(platform_dir.glob("**/*"))
                if files:
                    result.append(f"\n## {p.capitalize()} Files:")
                    for file in sorted(files):
                        if file.is_file():
                            rel_path = file.relative_to(base_dir)
                            result.append(f"- {rel_path}")
                else:
                    result.append(f"\n## {p.capitalize()}: No files created yet")
            else:
                result.append(f"\n## {p.capitalize()}: Directory not created yet")
                
        if not result:
            return "No files have been created yet."
            
        return "\n".join(result)
        
    except Exception as e:
        return f"Error listing files: {str(e)}"
        
@app_coder.tool
async def create_package_json(ctx: RunContext[AppCoderDeps], platform: str, dependencies: Dict[str, str] = None, dev_dependencies: Dict[str, str] = None, scripts: Dict[str, str] = None, name: str = None, version: str = "1.0.0") -> str:
    """
    Create a package.json file for a specific platform with appropriate dependencies.
    
    Args:
        ctx: The run context
        platform: Target platform (react, electron, nodejs, nativescript)
        dependencies: Dictionary of dependencies and versions
        dev_dependencies: Dictionary of dev dependencies and versions
        scripts: Dictionary of npm scripts
        name: Package name
        version: Package version
        
    Returns:
        str: Status message
    """
    try:
        # Create platform-specific directory if it doesn't exist
        base_dir = Path("workbench") / platform.lower()
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file path
        file_path = base_dir / "package.json"
        
        # Set default package name if not provided
        if not name:
            name = f"codeper-{platform.lower()}-app"
            
        # Create package.json content
        package_json = {
            "name": name,
            "version": version,
            "description": f"Codeper generated {platform} application",
            "main": get_main_file_for_platform(platform),
            "scripts": scripts or get_default_scripts_for_platform(platform),
            "dependencies": dependencies or get_default_dependencies_for_platform(platform),
            "devDependencies": dev_dependencies or get_default_dev_dependencies_for_platform(platform)
        }
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(package_json, f, indent=2)
            
        return f"Successfully created package.json for {platform} platform at {file_path}"
    
    except Exception as e:
        return f"Error creating package.json: {str(e)}"

def get_main_file_for_platform(platform: str) -> str:
    """Get the default main file for a specific platform."""
    platform_main_files = {
        "react": "src/index.js",
        "electron": "main.js",
        "nodejs": "server.js",
        "nativescript": "app.js"
    }
    return platform_main_files.get(platform.lower(), "index.js")

def get_default_scripts_for_platform(platform: str) -> Dict[str, str]:
    """Get default npm scripts for a specific platform."""
    scripts = {
        "react": {
            "start": "react-scripts start",
            "build": "react-scripts build",
            "test": "react-scripts test",
            "eject": "react-scripts eject"
        },
        "electron": {
            "start": "electron .",
            "build": "electron-builder",
            "pack": "electron-builder --dir"
        },
        "nodejs": {
            "start": "node server.js",
            "dev": "nodemon server.js",
            "test": "jest"
        },
        "nativescript": {
            "android": "ns run android",
            "ios": "ns run ios"
        }
    }
    return scripts.get(platform.lower(), {"start": "node index.js"})

def get_default_dependencies_for_platform(platform: str) -> Dict[str, str]:
    """Get default dependencies for a specific platform."""
    dependencies = {
        "react": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "react-scripts": "5.0.1"
        },
        "electron": {
            "electron-squirrel-startup": "^1.0.0"
        },
        "nodejs": {
            "express": "^4.18.2",
            "cors": "^2.8.5",
            "dotenv": "^16.3.1"
        },
        "nativescript": {
            "@nativescript/core": "^8.5.3",
            "@nativescript/theme": "^3.0.2"
        }
    }
    return dependencies.get(platform.lower(), {})

def get_default_dev_dependencies_for_platform(platform: str) -> Dict[str, str]:
    """Get default dev dependencies for a specific platform."""
    dev_dependencies = {
        "react": {
            "@testing-library/jest-dom": "^5.16.5",
            "@testing-library/react": "^13.4.0",
            "@testing-library/user-event": "^13.5.0"
        },
        "electron": {
            "electron": "^25.3.1",
            "electron-builder": "^24.4.0"
        },
        "nodejs": {
            "nodemon": "^2.0.22",
            "jest": "^29.5.0"
        },
        "nativescript": {
            "@nativescript/android": "^8.5.0",
            "@nativescript/ios": "^8.5.0",
            "@nativescript/webpack": "^5.0.15"
        }
    }
    return dev_dependencies.get(platform.lower(), {})

@app_coder.tool
async def get_code_example(ctx: RunContext[AppCoderDeps], query: str, platform: str) -> str:
    """
    Find and retrieve a code example for a specific platform based on the query.
    
    Args:
        ctx: The run context
        query: Search query describing what code example is needed
        platform: Target platform (react, electron, nodejs, nativescript)
        
    Returns:
        str: Code example with explanation
    """
    try:
        # Use the retrieve_relevant_documentation tool but focus on finding examples
        example_query = f"code example {query}"
        docs = await retrieve_relevant_documentation(ctx, example_query, platform)
        
        if "No relevant documentation found" in docs:
            # If no specific example found, provide a basic template
            return get_basic_template_for_platform(platform, query)
        
        return f"Code examples for {query} in {platform}:\n\n{docs}"
        
    except Exception as e:
        return f"Error finding code example: {str(e)}"

def get_basic_template_for_platform(platform: str, feature: str) -> str:
    """Provide a basic template for common features by platform."""
    if platform.lower() == "react":
        return f"""
# Basic React Component Template for {feature}

```jsx
import React, {{ useState, useEffect }} from 'react';

function {feature.replace(' ', '')}Component() {{
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {{
    // Fetch or initialize data for {feature}
    const fetchData = async () => {{
      try {{
        // Replace with actual data fetching
        const response = await fetch('/api/{feature.lower().replace(' ', '-')}');
        const result = await response.json();
        setData(result);
      }} catch (error) {{
        console.error('Error fetching data:', error);
      }} finally {{
        setLoading(false);
      }}
    }};

    fetchData();
  }}, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div className="{feature.lower().replace(' ', '-')}-container">
      <h2>{feature} Component</h2>
      {{/* Render your {feature} UI here */}}
    </div>
  );
}}

export default {feature.replace(' ', '')}Component;
```
"""
    elif platform.lower() == "electron":
        return f"""
# Basic Electron Template for {feature}

## main.js
```javascript
const {{ app, BrowserWindow, ipcMain }} = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {{
  mainWindow = new BrowserWindow({{
    width: 800,
    height: 600,
    webPreferences: {{
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }}
  }});

  mainWindow.loadFile('index.html');
  
  // Implement {feature} functionality here
  ipcMain.handle('{feature.lower().replace(' ', '-')}:action', async (event, args) => {{
    // Implement {feature} functionality
    return {{ success: true, data: {{ feature: '{feature}' }} }};
  }});
}}

app.whenReady().then(() => {{
  createWindow();
}});

app.on('window-all-closed', () => {{
  if (process.platform !== 'darwin') app.quit();
}});
```

## preload.js
```javascript
const {{ contextBridge, ipcRenderer }} = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {{
  {feature.lower().replace(' ', '')}: (args) => ipcRenderer.invoke('{feature.lower().replace(' ', '-')}:action', args)
}});
```

## renderer.js
```javascript
document.addEventListener('DOMContentLoaded', () => {{
  document.getElementById('{feature.lower().replace(' ', '-')}-button').addEventListener('click', async () => {{
    try {{
      const result = await window.electronAPI.{feature.lower().replace(' ', '')}({{ param: 'value' }});
      console.log('Result:', result);
    }} catch (error) {{
      console.error('Error:', error);
    }}
  }});
}});
```
"""
    elif platform.lower() == "nodejs":
        return f"""
# Basic Node.js Template for {feature}

## server.js
```javascript
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// {feature} Route
app.get('/api/{feature.lower().replace(' ', '-')}', (req, res) => {{
  try {{
    // Implement {feature} functionality here
    const data = {{
      feature: '{feature}',
      timestamp: new Date().toISOString()
    }};
    
    res.status(200).json(data);
  }} catch (error) {{
    console.error(`Error in {feature} endpoint:`, error);
    res.status(500).json({{ error: 'Internal server error' }});
  }}
}});

// Start server
app.listen(PORT, () => {{
  console.log(`Server running on port ${{PORT}}`);
}});
```
"""
    elif platform.lower() == "nativescript":
        return f"""
# Basic NativeScript Template for {feature}

## app.js
```javascript
import {{ Application }} from '@nativescript/core';

// Initialize {feature} functionality
console.log('Initializing {feature} feature');

Application.run({{ moduleName: 'app-root' }});
```

## app-root.xml
```xml
<Frame defaultPage="main-page"></Frame>
```

## main-page.js
```javascript
import {{ Observable }} from '@nativescript/core';

export function onNavigatingTo(args) {{
  const page = args.object;
  const viewModel = new Observable();
  
  viewModel.set('title', '{feature} Feature');
  viewModel.set('message', 'Welcome to the {feature} functionality');
  
  // Implement {feature} functionality
  viewModel.set('doAction', () => {{
    console.log('Performing {feature} action');
    // Add your implementation here
    viewModel.set('message', '{feature} action performed successfully!');
  }});
  
  page.bindingContext = viewModel;
}}
```

## main-page.xml
```xml
<Page xmlns="http://schemas.nativescript.org/tns.xsd" navigatingTo="onNavigatingTo">
    <ActionBar title="{{ title }}" />
    
    <StackLayout>
        <Label text="{{ message }}" class="h2 text-center m-10" />
        <Button text="Perform {feature} Action" tap="{{ doAction }}" class="btn btn-primary" />
    </StackLayout>
</Page>
```
"""
    else:
        return f"Platform {platform} not recognized or no template available for {feature}."

@app_coder.tool
async def create_readme(ctx: RunContext[AppCoderDeps], platforms: List[str] = None) -> str:
    """
    Create a README.md file with setup and usage instructions for all platforms.
    
    Args:
        ctx: The run context
        platforms: List of platforms to include (if None, use ctx.deps.platforms)
        
    Returns:
        str: Status message
    """
    try:
        if not platforms:
            platforms = ctx.deps.platforms
            
        # Create workbench directory if it doesn't exist
        workbench_dir = Path("workbench")
        workbench_dir.mkdir(parents=True, exist_ok=True)
        
        # Format platform names for display
        platform_display_names = {
            "react": "React (Web)",
            "electron": "Electron (Desktop)",
            "nodejs": "Node.js (Server)",
            "nativescript": "NativeScript (Mobile)"
        }
        
        # Get formatted platform names
        formatted_platforms = [platform_display_names.get(p.lower(), p) for p in platforms]
        platforms_str = ", ".join(formatted_platforms)
        
        # Build README content
        content = f"""# Codeper Generated Application

## Overview

This is a cross-platform application generated by Codeper, targeting the following platforms:
{", ".join([f"**{p}**" for p in formatted_platforms])}

## Project Structure

"""
        # Add platform-specific sections
        for platform in platforms:
            platform_dir = workbench_dir / platform.lower()
            if platform_dir.exists():
                content += f"""### {platform_display_names.get(platform.lower(), platform)} Structure

```
{platform.lower()}/
"""
                # List files for this platform
                files = sorted(platform_dir.glob("**/*"))
                if files:
                    for file in files:
                        if file.is_file():
                            rel_path = file.relative_to(platform_dir)
                            content += f"├── {rel_path}\n"
                content += "```\n\n"
            
        # Add setup instructions for each platform
        content += "## Setup Instructions\n\n"
        
        for platform in platforms:
            content += f"""### {platform_display_names.get(platform.lower(), platform)}

1. Navigate to the {platform.lower()} directory:
   ```
   cd {platform.lower()}
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. {get_run_instructions(platform)}

"""
        
        # Add common notes
        content += """## Additional Notes

- This application was generated by Codeper, an AI-powered cross-platform app development system.
- Modify the code as needed for your specific requirements.
- For any issues or questions, refer to the platform-specific documentation.
"""
        
        # Write content to README.md
        readme_path = workbench_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Successfully created README.md at {readme_path}"
    
    except Exception as e:
        return f"Error creating README: {str(e)}"

def get_run_instructions(platform: str) -> str:
    """Get platform-specific run instructions."""
    if platform.lower() == "react":
        return """Run the development server:
   ```
   npm start
   ```

   The application will be available at http://localhost:3000."""
    elif platform.lower() == "electron":
        return """Run the application:
   ```
   npm start
   ```
   
   To build a distributable package:
   ```
   npm run build
   ```"""
    elif platform.lower() == "nodejs":
        return """Start the server:
   ```
   npm start
   ```
   
   For development with auto-reload:
   ```
   npm run dev
   ```
   
   The API will be available at http://localhost:3000."""
    elif platform.lower() == "nativescript":
        return """Run on Android:
   ```
   npm run android
   ```
   
   Run on iOS:
   ```
   npm run ios
   ```
   
   Note: iOS builds require a Mac with Xcode installed."""
    else:
        return """Run the application:
   ```
   npm start
   ```"""

@app_coder.tool
async def create_gitignore(ctx: RunContext[AppCoderDeps]) -> str:
    """
    Create a .gitignore file in the workbench directory with appropriate patterns.
    
    Args:
        ctx: The run context
        
    Returns:
        str: Status message
    """
    try:
        # Create workbench directory if it doesn't exist
        workbench_dir = Path("workbench")
        workbench_dir.mkdir(parents=True, exist_ok=True)
        
        # Common gitignore patterns
        common_patterns = [
            "# Dependencies",
            "node_modules/",
            ".pnp/",
            ".pnp.js",
            
            "# Testing",
            "coverage/",
            
            "# Production",
            "build/",
            "dist/",
            
            "# Misc",
            ".DS_Store",
            ".env.local",
            ".env.development.local",
            ".env.test.local",
            ".env.production.local",
            
            "# Logs",
            "npm-debug.log*",
            "yarn-debug.log*",
            "yarn-error.log*",
            
            "# Editor directories and files",
            ".idea/",
            ".vscode/",
            "*.suo",
            "*.ntvs*",
            "*.njsproj",
            "*.sln",
            "*.sw?"
        ]
        
        # Platform-specific gitignore patterns
        platform_patterns = {
            "react": [
                "# React specific",
                "/node_modules",
                "/.pnp",
                ".pnp.js",
                "/coverage",
                "/build"
            ],
            "electron": [
                "# Electron specific",
                "out/",
                "dist/",
                ".webpack/"
            ],
            "nodejs": [
                "# Node.js specific",
                ".env",
                ".npm",
                "logs",
                "*.log",
                "pids",
                "*.pid",
                "*.seed",
                "*.pid.lock"
            ],
            "nativescript": [
                "# NativeScript specific",
                "hooks/",
                "platforms/",
                "node_modules/",
                "app/**/*.js",
                "!app/tns_modules/**/*.js",
                "report/",
                ".migration_backup/"
            ]
        }
        
        # Build content based on selected platforms
        content = "# Generated by Codeper\n\n"
        content += "\n".join(common_patterns) + "\n"
        
        # Add platform-specific patterns
        for platform in ctx.deps.platforms:
            if platform.lower() in platform_patterns:
                content += f"\n# {platform.capitalize()} specific\n"
                content += "\n".join(platform_patterns[platform.lower()]) + "\n"
        
        # Write content to .gitignore
        gitignore_path = workbench_dir / ".gitignore"
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Successfully created .gitignore at {gitignore_path}"
    
    except Exception as e:
        return f"Error creating .gitignore: {str(e)}"

@app_coder.tool
async def create_env_example(ctx: RunContext[AppCoderDeps]) -> str:
    """
    Create a .env.example file with placeholder environment variables.
    
    Args:
        ctx: The run context
        
    Returns:
        str: Status message
    """
    try:
        # Create workbench directory if it doesn't exist
        workbench_dir = Path("workbench")
        workbench_dir.mkdir(parents=True, exist_ok=True)
        
        # Common environment variables
        common_vars = [
            "# Common environment variables",
            "NODE_ENV=development"
        ]
        
        # Platform-specific environment variables
        platform_vars = {
            "react": [
                "# React environment variables",
                "REACT_APP_API_URL=http://localhost:3000/api",
                "REACT_APP_ENV=development"
            ],
            "electron": [
                "# Electron environment variables",
                "ELECTRON_START_URL=http://localhost:3000"
            ],
            "nodejs": [
                "# Node.js environment variables",
                "PORT=3000",
                "DATABASE_URL=postgres://username:password@localhost:5432/database",
                "JWT_SECRET=your_jwt_secret_here"
            ],
            "nativescript": [
                "# NativeScript environment variables",
                "API_URL=http://localhost:3000/api"
            ]
        }
        
        # Build content based on selected platforms
        content = "# Environment Variables - Copy to .env and fill in your values\n\n"
        content += "\n".join(common_vars) + "\n"
        
        # Add platform-specific variables
        for platform in ctx.deps.platforms:
            if platform.lower() in platform_vars:
                content += f"\n{platform_vars[platform.lower()][0]}\n"
                content += "\n".join(platform_vars[platform.lower()][1:]) + "\n"
        
        # Write content to .env.example
        env_path = workbench_dir / ".env.example"
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Successfully created .env.example at {env_path}"
    
    except Exception as e:
        return f"Error creating .env.example: {str(e)}"

@app_coder.tool
async def scaffold_project_structure(ctx: RunContext[AppCoderDeps], platform: str) -> str:
    """
    Create a basic project structure for a specific platform.
    
    Args:
        ctx: The run context
        platform: Target platform (react, electron, nodejs, nativescript)
        
    Returns:
        str: Status message
    """
    try:
        # Create platform-specific directory structure
        base_dir = Path("workbench") / platform.lower()
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure based on platform
        if platform.lower() == "react":
            (base_dir / "public").mkdir(exist_ok=True)
            (base_dir / "src").mkdir(exist_ok=True)
            (base_dir / "src" / "components").mkdir(exist_ok=True)
            (base_dir / "src" / "pages").mkdir(exist_ok=True)
            (base_dir / "src" / "services").mkdir(exist_ok=True)
            (base_dir / "src" / "hooks").mkdir(exist_ok=True)
            (base_dir / "src" / "assets").mkdir(exist_ok=True)
            
            # Create minimal index.html
            with open(base_dir / "public" / "index.html", "w", encoding="utf-8") as f:
                f.write("""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Codeper generated React application" />
    <title>React App</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
""")
            
            # Create minimal index.js
            with open(base_dir / "src" / "index.js", "w", encoding="utf-8") as f:
                f.write("""import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""")
            
        elif platform.lower() == "electron":
            # Create index.html
            with open(base_dir / "index.html", "w", encoding="utf-8") as f:
                f.write("""<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <title>Electron App</title>
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'" />
    <meta http-equiv="X-Content-Security-Policy" content="default-src 'self'; script-src 'self'" />
    <link rel="stylesheet" type="text/css" href="styles.css" />
  </head>
  <body>
    <h1>Electron App</h1>
    <p>Welcome to your Electron application.</p>
    <script src="./renderer.js"></script>
  </body>
</html>
""")
            
            # Create empty styles.css
            with open(base_dir / "styles.css", "w", encoding="utf-8") as f:
                f.write("""body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  margin: 0;
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

h1 {
  color: #333;
}
""")
            
            # Create empty renderer.js
            with open(base_dir / "renderer.js", "w", encoding="utf-8") as f:
                f.write("""// This file is executed in the renderer process
document.addEventListener('DOMContentLoaded', () => {
  console.log('Renderer process started');
});
""")
            
        elif platform.lower() == "nodejs":
            (base_dir / "routes").mkdir(exist_ok=True)
            (base_dir / "controllers").mkdir(exist_ok=True)
            (base_dir / "models").mkdir(exist_ok=True)
            (base_dir / "middleware").mkdir(exist_ok=True)
            (base_dir / "utils").mkdir(exist_ok=True)
            
            # Create basic index route
            with open(base_dir / "routes" / "index.js", "w", encoding="utf-8") as f:
                f.write("""const express = require('express');
const router = express.Router();

router.get('/', (req, res) => {
  res.status(200).json({ message: 'API is working!' });
});

module.exports = router;
""")
            
        elif platform.lower() == "nativescript":
            (base_dir / "app").mkdir(exist_ok=True)
            (base_dir / "app" / "views").mkdir(exist_ok=True)
            (base_dir / "app" / "services").mkdir(exist_ok=True)
            
            # Create app-root.xml
            with open(base_dir / "app" / "app-root.xml", "w", encoding="utf-8") as f:
                f.write("""<Frame defaultPage="views/main-page"></Frame>
""")
            
            # Create main page
            (base_dir / "app" / "views").mkdir(exist_ok=True)
            with open(base_dir / "app" / "views" / "main-page.xml", "w", encoding="utf-8") as f:
                f.write("""<Page xmlns="http://schemas.nativescript.org/tns.xsd" navigatingTo="onNavigatingTo">
  <ActionBar title="Home" />
  <StackLayout>
    <Label text="Welcome to NativeScript" class="h2 text-center m-10" />
  </StackLayout>
</Page>
""")
            
            with open(base_dir / "app" / "views" / "main-page.js", "w", encoding="utf-8") as f:
                f.write("""import { Observable } from '@nativescript/core';

export function onNavigatingTo(args) {
  const page = args.object;
  const viewModel = new Observable();
  
  page.bindingContext = viewModel;
}
""")
        
        # Create package.json for this platform
        await create_package_json(ctx, platform)
        
        return f"Successfully scaffolded {platform} project structure at {base_dir}"
    
    except Exception as e:
        return f"Error scaffolding project structure: {str(e)}"