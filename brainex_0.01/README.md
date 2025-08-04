# RAG Agent with Gemini Integration

A Retrieval-Augmented Generation (RAG) agent that uses Rule Based Reasoning to provide intelligent responses by searching both a local knowledge base (Firestore) and the web.

## 🚀 Features

- **Intelligent RAG System**: Combines local knowledge base with web search
- **Firestore Database**: Stores and retrieves knowledge from Google Cloud Firestore
- **Web Search**: Integrates with Serper API for real-time web research
- **Tool-based Architecture**: Modular design with LangGraph for complex workflows
- **Conversation Memory**: Maintains conversation history across sessions

## 📋 Prerequisites

- Python 3.8+
- Google Cloud Project with Firestore enabled
- Gemini API key
- Serper API key (optional, for web search)

## 🏗️ Architecture

### Components

1. **RAGAgent**: Main agent class that orchestrates the workflow
2. **Tools**:
   - `SearchFirestore`: Searches local knowledge base
   - `WebSearch`: Performs web research via Serper API
   - `SaveKnowledge`: Saves new information to Firestore
3. **LangGraph Workflow**: Manages the conversation flow
4. **Gemini LLM**: Processes queries and generates responses

### Workflow

1. **Query Processing**: User query is received
2. **Knowledge Search**: Searches Firestore for existing information
3. **Web Research**: If needed, searches the web for current information
4. **Knowledge Storage**: Saves new information for future use
5. **Response Generation**: Generates a response

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   SERPER_API_KEY=your_serper_api_key_here
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/firestore-credentials.json
   ```

4. **Set up Google Cloud credentials**:
   - Download your Firestore service account key
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable


## 🚀 Usage

### Basic Usage

```python
from debug import RAGAgent

# Initialize the agent
agent = RAGAgent()

# Ask a question
response = agent.query("What is quantum computing?")
print(response)
```

### Advanced Usage

```python
# Use with conversation history
response1 = agent.query("Tell me about machine learning", thread_id="ml_session")
response2 = agent.query("What are the latest developments?", thread_id="ml_session")

# Get conversation history
history = agent.get_conversation_history(thread_id="ml_session")

# Stream responses
for chunk in agent.stream_query("Explain AI trends", thread_id="stream_test"):
    print(chunk, end="")
```

### Running the Main Example

```bash
python debug.py
```

This will run several test queries to demonstrate the system's capabilities.




