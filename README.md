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
    ```sh
    git clone https://github.com/yourusername/claudecord.git
    cd claudecord
    ```

2. Create and activate a virtual environment:
    ```sh
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux/MacOS
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Create configuration files:
    ```sh
    # Create .env file
    touch .env

    # Create config directory
    mkdir config
    ```

5. Configure environment variables in `.env`:
    ```sh
    DISCORD_TOKEN=your_discord_bot_token
    ANTHROPIC_API_KEY=your_anthropic_api_key
    DB_PATH=data/conversations.db
    BACKUP_PATH=backups
    LOG_LEVEL=INFO
    COMMAND_PREFIX=>
    ADMIN_ROLE=ClaudeAdmin
    MOD_ROLE=ClaudeMod
    ```

6. Create database directory:
    ```sh
    mkdir data
    ```

### Step 3: Configuration

1. Create `config/config.yaml`:
    ```yaml
    bot:
      command_prefix: ">"
      description: "A Discord bot powered by Claude 3.5 Sonnet"
      status_message: "Ready to chat!"
      color_theme: 0xda7756

    claude:
      model: "claude-3-sonnet-20240229"
      max_tokens: 4096
      temperature: 0.7
      max_memory: 20
      system_prompt: |
        You are a helpful AI assistant. You provide clear, accurate, and engaging responses 
        while maintaining a friendly tone. When dealing with technical topics, you explain 
        concepts thoroughly but accessibly.

    database:
      path: "data/conversations.db"
      backup_interval: 86400  # 24 hours
      max_attachment_size: 100000000  # 100MB
      cleanup_interval: 604800  # 7 days

    limits:
      max_conversation_length: 50
      rate_limit: 5  # messages per second
      max_tokens_per_request: 4096
      max_image_size: 20000000  # 20MB
    ```

### Step 4: Running the Bot

1. Start the bot:
    ```sh
    # Windows
    python main.py

    # Linux/MacOS
    python3 main.py
    ```

2. Verify the bot is running:
- Check console for startup messages
- Bot should appear online in Discord
- Try the test command: `>ping`

### Step 5: Setting Up Permissions

1. Create roles in your Discord server:
- ClaudeAdmin: Full access to all commands
- ClaudeMod: Moderation commands access

2. Assign roles to trusted users

### Step 6: Basic Usage

1. Start a conversation:
    ```
    >ask What can you help me with?
    ```

2. Configure preferences:
    ```
    >style formal
    >language en
    >voice professional
    ```

3. View bot status:
    ```
    >status
    >health
    ```

### Monitoring and Maintenance

1. Check logs:
- Located in `logs/bot.log`
- Contains detailed operation information

2. Database backups:
- Automatic backups in `backups/` directory
- Manual backup: `>backup`

3. Performance monitoring:
- Use `>health` for system status
- Monitor `data/metrics.log` for performance data

### Troubleshooting

1. Bot not responding:
- Check bot token in `.env`
- Verify bot permissions
- Check console for error messages

2. Database issues:
- Ensure write permissions for data directory
- Check disk space
- Verify SQLite installation

3. Rate limiting:
- Adjust limits in config.yaml
- Monitor rate limit hits with `>health`

4. Memory issues:
- Reduce max_memory in config
- Monitor memory usage with `>health`
- Consider cleaning old conversations

### Security Recommendations

1. File permissions:
    ```sh
    chmod 600 .env
    chmod 600 config/config.yaml
    ```

2. Regular updates:
    ```sh
    git pull
    pip install -r requirements.txt --upgrade
    ```

3. Backup strategy:
- Enable automatic backups
- Store backups securely
- Test restoration periodically

### Advanced Configuration

For detailed configuration options and advanced features, see the [Wiki](https://github.com/yourusername/claudecord/wiki).

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/yourusername/claudecord/issues) page.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.