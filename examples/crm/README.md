# CRM Agent with Slack Integration

This directory contains the CRM agent with full Slack integration capabilities, allowing real-time communication and memory management through Slack channels.

## Overview

The CRM Slack integration provides:
- Real-time message processing from Slack
- Memory storage and retrieval through Slack commands
- Interactive responses with context awareness
- Thread management and conversation tracking
- Webhook-based event handling

## Architecture

```
crm/
├── crm_server.py          # Main CRM FastAPI server
├── query_constructor.py   # CRM-specific query constructor
├── slack_server.py        # Slack integration server
├── slack_service.py       # Slack service utilities
└── README.md             # This file
```

## Running the CRM Agent
- Follow the below steps for making a Slack App
- Install the App into your chosen channel
- Run python crm_server.py and python slack_server.py
- Configure ngrok for TLS issues
- Start the MemMachine backend (see MemMachine README.md)

## Prerequisites

- Python 3.12+
- FastAPI and dependencies
- MemMachine backend running
- Slack app with bot permissions
- ngrok for local development (free tier available)

## Slack Bot Setup

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App"
3. Choose "From scratch"
4. Enter app name (e.g., "CRM Memory Bot")
5. Select your workspace

### 2. Configure Bot Permissions

Navigate to **OAuth & Permissions** and ensure you have these scopes:

#### Bot Token Scopes 
```
app_mentions:read          # View messages that directly mention @CRM Bot
channels:history           # View messages in public channels
channels:read              # View basic information about public channels
chat:write                 # Send messages as @CRM Bot
groups:history             # View messages in private channels
groups:read                # View basic information about private channels
users:read                 # View people in a workspace
```

#### Additional Scopes (Optional)
If you want to expand functionality, consider adding:
```
commands                   # Add slash commands
files:read                 # Read files shared in channels
reactions:read             # Read reactions
reactions:write            # Add/remove reactions
team:read                  # View workspace name
users:read.email           # View email addresses
```

### 3. Install App to Workspace

1. Go to **OAuth & Permissions**
2. Click "Install to Workspace"
3. Review permissions and click "Allow"
4. Copy the **Bot User OAuth Token** (starts with `xoxb-`)

### 4. Configure Event Subscriptions

1. Go to **Event Subscriptions**
2. Enable Events: **On**
3. Request URL: `https://your-ngrok-url.ngrok.io/slack/events`
4. Subscribe to Bot Events:
   ```
   app_mention          # When @CRM Bot is mentioned
   message.channels     # Messages in public channels
   message.groups       # Messages in private channels
   ```

## Local Development Setup

### 1. Install Dependencies

```bash
pip install fastapi uvicorn httpx python-dotenv
```

### 2. Set Environment Variables

Create a `.env` file in the crm directory:

```bash
# Slack Configuration
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_SIGNING_SECRET=your-signing-secret-here
SLACK_APP_TOKEN=xapp-your-app-token-here

# MemMachine Backend
MEMORY_BACKEND_URL=http://localhost:8080

# Server Configuration
SLACK_SERVER_PORT=8001
LOG_LEVEL=INFO
```

### 3. Install and Configure ngrok

#### Install ngrok
```bash
# macOS (using Homebrew)
brew install ngrok

# Or download from https://ngrok.com/download
```

#### Start ngrok
```bash
# Start ngrok on port 8001 (or your chosen port)
ngrok http 8001

# Note the HTTPS URL (e.g., https://abc123.ngrok.io)
```

#### Update Slack App Configuration
1. Copy the ngrok HTTPS URL
2. Update your Slack app's Event Subscriptions URL
3. Update any slash command URLs

### 4. Run the Slack Server

```bash
cd agents/crm
python slack_server.py
```

The server will start on `http://localhost:8001` and be accessible via ngrok.


## API Endpoints

### Slack Events
- `POST /slack/events` - Handle Slack events and interactions

### Slack Commands
- `POST /slack/commands` - Handle slash commands

