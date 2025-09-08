# Intelligent Memory Agents

This directory contains specialized AI agents that integrate with the MemMachine system. Each agent is designed to handle specific domains and use cases, providing tailored query construction and memory management capabilities.

## Overview

The agents system is built on a modular architecture with:
- **Base Query Constructor**: Abstract base class for all query constructors
- **Specialized Agents**: Domain-specific implementations (CRM, Financial Analyst, etc.)
- **FastAPI Servers**: RESTful APIs for each agent
- **Slack Integration**: Real-time communication capabilities

## Architecture

```
agents/
├── base_query_constructor.py      # Base class for query constructors
├── default_query_constructor.py   # Default/general-purpose query constructor
├── example_server.py             # Example FastAPI server implementation
├── crm/                          # CRM-specific agent
│   ├── crm_server.py            # CRM FastAPI server
│   ├── query_constructor.py     # CRM query constructor
│   ├── slack_server.py          # Slack integration for CRM
│   └── slack_service.py         # Slack service utilities
└── financial_analyst/            # Financial analysis agent
    ├── financial_server.py      # Financial analyst FastAPI server
    └── query_constructor.py     # Financial query constructor
```

## Connecting to MemMachine

Start MemMachine by either running the Python file or the Docker container. These example agents all use the REST API from memmachine's app.py, but you can also integrate using the MCP server.

## Available Agents

### 1. Default Agent (`example_server.py`)
- **Purpose**: General-purpose AI assistant for any chatbot
- **Port**: 8000 (configurable via `EXAMPLE_SERVER_PORT`)
- **Features**: Basic memory storage and retrieval
- **Use Case**: General conversations and information management

### 2. CRM Agent (`crm/`)
- **Purpose**: Customer Relationship Management
- **Port**: 8000 (configurable via `CRM_PORT`)
- **Features**: 
  - Customer data management
  - Sales pipeline tracking
  - Slack integration for real-time communication
  - CRM-specific query construction
- **Use Case**: Sales teams, customer support, relationship management

### 3. Financial Analyst Agent (`financial_analyst/`)
- **Purpose**: Financial analysis and reporting
- **Port**: 8000 (configurable via `FINANCIAL_PORT`)
- **Features**:
  - Financial data analysis
  - Investment insights
  - Market trend analysis
  - Financial reporting
- **Use Case**: Financial advisors, investment teams, accounting departments

### 4. Streamlit Frontend (`frontend/`)
- **Purpose**: Web-based testing interface for all agents
- **Port**: 8502 (configurable via Streamlit default)
- **Features**:
  - Interactive web UI for testing agents
  - Memory management interface
  - Real-time conversation testing
  - Model selection and configuration
  - Persona-based testing
- **Use Case**: Development, testing, and demonstration of agent capabilities

## Quick Start

### Prerequisites
- Python 3.12+
- FastAPI
- Requests library
- MemMachine backend running
- Environment variables configured

### Running an Agent

1. **Set up environment variables**:
   ```bash
   MEMORY_BACKEND_URL="http://localhost:8080"
   OPENAI_API_KEY="your-openai-api-key"
   ```

2. **Run a specific agent**:
   ```bash
   # Default agent
   python example_server.py
   
   # CRM agent
   cd crm
   python crm_server.py
   
   # Financial analyst agent
   cd financial_analyst
   python financial_server.py
   ```

3. **Access the API**:
   - Default: `http://localhost:8000`
   - CRM: `http://localhost:8000` (when running CRM server)
   - Financial: `http://localhost:8000` (when running Financial server)
   - Frontend: `http://localhost:8502` (when running Streamlit app)

## Using the Streamlit Frontend for Testing

The Streamlit frontend provides an interactive web interface for testing all agents and their memory capabilities.

### Starting the Frontend

1. **Prerequisites**:
   - MemMachine backend running (see main README)
   - At least one agent server running (CRM, Financial, or Default)
   - Required environment variables set

2. **Run the frontend**:
   ```bash
   cd agents/frontend
   streamlit run app.py
   ```

3. **Access the interface**:
   - Open your browser to `http://localhost:8502`

### Frontend Features

#### Model Configuration
- **Model Selection**: Choose from various LLM providers (OpenAI, Anthropic, DeepSeek, Meta, Mistral)
- **API Key Management**: Configure API keys for different providers
- **Model Parameters**: Adjust temperature, max tokens, and other settings

#### Memory Testing
- **Persona Management**: Create and manage different user personas
- **Memory Storage**: Test memory storage and retrieval
- **Context Search**: Search through stored memories
- **Profile Management**: View and manage user profiles

#### Agent Testing
- **Real-time Chat**: Test conversations with different agents
- **Memory Integration**: See how agents use stored memories
- **Response Analysis**: Compare responses with and without memory context
- **Rationale Display**: View how personas influence responses

### Testing Workflow

1. **Start Services**:
   ```bash
   # Terminal 1: Start MemMachine backend
   cd memmachine/src
   python -m server.app
   
   # Terminal 2: Start an agent (e.g., CRM)
   cd agents/crm
   python crm_server.py
   
   # Terminal 3: Start the frontend
   cd agents/frontend
   streamlit run app.py
   ```

2. **Configure the Frontend**:
   - Set the CRM Server URL (default: `http://localhost:8000`)
   - Select your preferred model and provider
   - Enter your API key

3. **Test Memory Operations**:
   - Create a new persona or use existing ones
   - Send messages to test memory storage
   - Use search functionality to retrieve memories
   - Test different conversation patterns

4. **Analyze Results**:
   - View memory storage logs
   - Compare responses with/without memory context
   - Check persona influence on responses

### Environment Variables for Frontend

```bash
# Required for frontend functionality
CRM_SERVER_URL=http://localhost:8000
MODEL_API_KEY=your-openai-api-key
OPENAI_API_KEY=your-openai-api-key

# Optional: For other providers
ANTHROPIC_API_KEY=your-anthropic-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

### Troubleshooting Frontend Issues

#### Common Issues:
1. **Connection Refused**: Ensure the agent server is running
2. **API Key Errors**: Verify your API keys are correct
3. **Memory Not Storing**: Check MemMachine backend is running
4. **Model Not Responding**: Verify model selection and API key

#### Debug Mode:
```bash
# Run with debug logging
LOG_LEVEL=DEBUG streamlit run app.py
```

### Frontend Architecture

The frontend consists of:
- **app.py**: Main Streamlit application
- **llm.py**: LLM integration and chat functionality
- **gateway_client.py**: API client for agent communication
- **model_config.py**: Model configuration and provider mapping
- **styles.css**: Custom styling for the interface

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_BACKEND_URL` | URL of the MemMachine backend service | `http://localhost:8080` |
| `OPENAI_API_KEY` | OpenAI API key for LLM access | Required |
| `EXAMPLE_SERVER_PORT` | Port for example server | `8000` |
| `CRM_PORT` | Port for CRM server | `8000` |
| `FINANCIAL_PORT` | Port for financial analyst server | `8000` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

### MemMachine Integration

All agents integrate with the MemMachine backend by:
1. Storing conversation episodes as memories
2. Retrieving relevant context for queries
3. Using profile information for personalized responses
4. Maintaining conversation history and context

## Query Constructor System

### Base Query Constructor
The `BaseQueryConstructor` class provides the foundation for all query constructors:

```python
class BaseQueryConstructor:
    def create_query(self, **kwargs) -> str:
        # Must be implemented by subclasses
        raise NotImplementedError
```

### Specialized Constructors

Each agent implements its own query constructor with domain-specific logic:

- **CRMQueryConstructor**: Optimized for customer relationship management
- **FinancialAnalystQueryConstructor**: Specialized for financial analysis
- **DefaultQueryConstructor**: General-purpose query handling

## Slack Integration

The CRM agent includes Slack integration for real-time communication:

### Features
- Real-time message processing
- Webhook handling
- Interactive responses
- Thread management

### Setup
1. Configure Slack app with webhook URL
2. Set up environment variables for Slack
3. Deploy the slack_server.py endpoint

## Development

### Adding a New Agent

1. **Create agent directory**:
   ```bash
   mkdir agents/new_agent
   cd agents/new_agent
   ```

2. **Implement query constructor**:
   ```python
   from base_query_constructor import BaseQueryConstructor
   
   class NewAgentQueryConstructor(BaseQueryConstructor):
       def create_query(self, **kwargs) -> str:
           # Implement domain-specific logic
           pass
   ```

3. **Create FastAPI server**:
   ```python
   from fastapi import FastAPI
   from query_constructor import NewAgentQueryConstructor
   
   app = FastAPI(title="New Agent Server")
   constructor = NewAgentQueryConstructor()
   
   # Implement endpoints
   ```

4. **Add configuration**:
   - Environment variables
   - Port configuration
   - MemMachine backend integration

### Testing

Each agent can be tested independently:

```bash
# Test memory storage
curl -X POST "http://localhost:8000/memory" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "query": "Hello world"}'

# Test query processing
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "query": "What did I say earlier?"}'
```

## Troubleshooting

### Common Issues

1. **MemMachine Backend Connection Error**:
   - Ensure the MemMachine backend is running on the correct port
   - Check `MEMORY_BACKEND_URL` environment variable

2. **OpenAI API Errors**:
   - Verify `OPENAI_API_KEY` is set correctly
   - Check API key permissions and quotas

3. **Port Conflicts**:
   - Ensure only one agent runs on each port
   - Use different ports for multiple agents

4. **Import Errors**:
   - Check Python path configuration
   - Ensure all dependencies are installed

### Logging

All agents support configurable logging:

```bash
LOG_LEVEL=DEBUG  # For detailed debugging
LOG_LEVEL=INFO   # For normal operation
LOG_LEVEL=ERROR  # For error-only logging
```

## Contributing

When adding new agents or features:

1. Follow the existing architecture patterns
2. Implement proper error handling
3. Add comprehensive logging
4. Include API documentation
5. Test with the MemMachine backend integration

## License

This project is part of the Intelligent Memory system. Please refer to the main project license for usage terms.