# ClaudeCord

A powerful Discord bot powered by Claude 3.5 Sonnet with advanced features and management capabilities.

## Detailed Setup Guide

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git
- A Discord account with developer access
- An Anthropic API key

### Step 1: Discord Bot Setup
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and name your bot
3. Go to the "Bot" section
4. Click "Add Bot"
5. Enable these Privileged Gateway Intents:
   - Presence Intent
   - Server Members Intent
   - Message Content Intent
6. Copy your bot token (you'll need this later)
7. Go to OAuth2 > URL Generator
8. Select these scopes:
   - bot
   - applications.commands
9. Select these bot permissions:
   - Read Messages/View Channels
   - Send Messages
   - Embed Links
   - Attach Files
   - Read Message History
   - Add Reactions
10. Copy the generated URL and use it to invite the bot to your server

### Step 2: Local Setup
1. Clone the repository:
