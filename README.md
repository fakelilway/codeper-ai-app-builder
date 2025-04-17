# Codeper: AI-Powered Cross-Platform App Development

Codeper is an intelligent system that automates the creation of cross-platform applications using generative AI and framework-specific knowledge retrieval. With Codeper, you can transform app ideas into functional code for web, desktop, and mobile platforms - even without deep technical expertise.

## 🚀 Features

- **Cross-Platform Development**: Generate code for multiple platforms simultaneously:
  - Web apps (React)
  - Desktop apps (Electron)
  - Mobile apps (NativeScript)
  - Server-side apps (Node.js)

- **Intelligent Architecture Design**: Analyzes requirements to create optimal app architecture with appropriate components and data flow between them.

- **Documentation-Aware Code Generation**: Uses Retrieval-Augmented Generation (RAG) to incorporate best practices from official framework documentation into generated code.

- **Complete Project Structure**: Creates fully-structured projects with package configurations, folder organization, and essential files.

- **User-Friendly Interface**: Streamlit-based UI with intuitive chat interface for describing app requirements and viewing generated code.

## 💻 System Architecture

Codeper consists of several key components working together:

1. **Reasoning Engine**: Powered by Deepseek's LLM to analyze requirements and generate development plans.

2. **App Coder**: Generates platform-specific code using OpenAI models and documentation-aware context.

3. **Documentation RAG**: Retrieves relevant framework documentation to inform code generation.

4. **Workflow Orchestrator**: Built with LangGraph to manage the multi-agent workflow.

5. **User Interface**: Streamlit application for user interaction and displaying results.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for testing generated code)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/fakelilway/codeper-ai-app-builder.git
   cd codeper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the following variables:
   ```
   # OpenAI API Configuration
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_BASE_URL=https://api.openai.com/v1

   # Deepseek API Configuration
   DEEPSEEK_API_KEY=your_deepseek_api_key
   DEEPSEEK_BASE_URL=https://api.deepseek.com

   # Supabase Configuration
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_key
   
   # Models Configuration
   REASONER_MODEL=deepseek-reasoner
   PRIMARY_MODEL=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-small
   ```

## 🚀 Running Codeper

Launch the application with:

```bash
python crawl_electron_docs.py

python crawl_nativescript_docs.py

python crawl_nodejs_docs.py

python crawl_react_docs.py

streamlit run streamlit_ui.py
```

The application will be available at http://localhost:8501 by default.

## 🧪 How to Use

1. **Describe Your App**: Enter a description of the app you want to build in the chat interface.

2. **Platform Selection**: The system will automatically determine which platforms are appropriate for your app.

3. **Architecture Design**: Codeper will analyze your requirements and design an appropriate architecture.

4. **Code Generation**: Platform-specific code will be generated for all selected platforms.

5. **View & Export**: Browse generated files in the sidebar and use them in your development workflow.

## 📚 Technical Details

### LLM Integration

Codeper uses two main language models:
- **Deepseek Reasoner**: For high-level app architecture and planning
- **OpenAI GPT-4o-mini**: For code generation and user interaction

### Documentation RAG

The system retrieves framework documentation from a Supabase vector database to ensure:
- Compliance with framework best practices
- Up-to-date API usage
- Proper component integration

### Generated Project Structure

For each platform, Codeper creates a complete project structure with:
- Entry point files
- Component/module files
- Configuration files (package.json, etc.)
- Basic documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- This project builds upon [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- Powered by [Deepseek](https://deepseek.com) and [OpenAI](https://openai.com) language models
- Documentation storage and retrieval via [Supabase](https://supabase.com)
