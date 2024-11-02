import discord
from discord import Intents, Message, Embed, Activity, ActivityType, Status
from discord.ext import commands
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from typing import Final, List, Dict
import asyncio
import logging
import os
from io import BytesIO
import yaml
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import time
from dataclasses import dataclass
from typing import Optional
import sys

from conversation_mem import ConversationStorage
from multimodal import process_file
from utils.permissions import PermissionManager, PermissionLevel
from utils.queue_manager import MessageQueue
from utils.themes import ThemeManager, Theme
from utils.components import InteractiveView, ConfirmationView, PaginationView, SettingsView
from utils.context_manager import ContextManager
from utils.embed_manager import EmbedManager, FormatStyle
from utils.monitoring import MonitoringSystem
from utils.input_sanitizer import InputSanitizer

# logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load environment variables
load_dotenv()

# Constants
DISCORD_TOK: Final[str] = os.getenv('DISCORD_TOKEN')
CLAUDE_KEY: Final[str] = os.getenv('ANTHROPIC_API_KEY')
MODEL_NAME: Final[str] = "claude-3-sonnet-20240229"
MAX_TOKENS: Final[int] = 4096
TEMPERATURE: Final[float] = 0.7
MAX_MEMORY: Final[int] = 20
SYSTEM_PROMPT: Final[str] = """
You are a helpful AI assistant. You provide clear, accurate, and engaging responses 
while maintaining a friendly tone. When dealing with technical topics, you explain concepts thoroughly 
but accessibly.
""".strip()

CONFIG_PATH = Path("config.yaml")
DEFAULT_CONFIG = {
    "model": "claude-3-sonnet-20240229",  # Updated to latest Sonnet model
    "max_tokens": 4096,
    "temperature": 0.7,
    "max_memory": 20,
    "system_prompt": """You are a helpful AI assistant. You provide clear, accurate, and engaging responses 
    while maintaining a friendly tone. When dealing with technical topics, you explain concepts thoroughly 
    but accessibly.""".strip()
}

@dataclass
class EnvConfig:
    discord_token: str
    claude_key: str
    database_path: str = "conversations.db"
    backup_path: str = "backups"
    log_level: str = "INFO"
    command_prefix: str = ">"
    admin_role: str = "ClaudeAdmin"
    mod_role: str = "ClaudeMod"

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

class ClaudeCordBot:
    def __init__(self):
        self.env = self._validate_env()
        self.config = self._load_config()
        self.intents = Intents.default()
        self.intents.message_content = True
        self.bot = commands.Bot(command_prefix=self.env.command_prefix, intents=self.intents)
        self.claude_client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.storage = ConversationStorage(self.env.database_path)
        self.theme_manager = ThemeManager()
        self.embed_manager = EmbedManager()
        self.context_manager = ContextManager(self.storage)
        self.setup_commands()
        self.message_queue = MessageQueue(rate_limit=self.config['limits']['rate_limit'])
        self.processing_lock = asyncio.Lock()
        self.rate_limit = 5  # messages per second
        self.last_processed = 0
        self.permissions = PermissionManager(self.env.admin_role, self.env.mod_role)
        self.bot.on_command_error = self.handle_error
        self.monitoring = MonitoringSystem()
        self.sanitizer = InputSanitizer()
        self.status_index = 0
        self.status_messages = [
            ("with neural networks 🧠", ActivityType.playing),
            ("your conversations 💭", ActivityType.watching),
            ("to your requests 👂", ActivityType.listening),
            ("Claude 3.5 Sonnet 🤖", ActivityType.competing),
            ("Need help? Mention me! ✨", ActivityType.custom),
        ]
        self._status_task = None

    def _validate_env(self) -> EnvConfig:
        """Validate and load environment variables"""
        required_vars = {
            'DISCORD_TOKEN': os.getenv('DISCORD_TOKEN'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        }
        
        # Check required vars
        missing = [k for k, v in required_vars.items() if not v]
        if missing:
            raise ConfigurationError(f"Missing required environment variables: {', '.join(missing)}")
            
        # Create EnvConfig with validated values
        return EnvConfig(
            discord_token=required_vars['DISCORD_TOKEN'],
            claude_key=required_vars['ANTHROPIC_API_KEY'],
            database_path=os.getenv('DB_PATH', 'conversations.db'),
            backup_path=os.getenv('BACKUP_PATH', 'backups'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            command_prefix=os.getenv('COMMAND_PREFIX', '>'),
            admin_role=os.getenv('ADMIN_ROLE', 'ClaudeAdmin'),
            mod_role=os.getenv('MOD_ROLE', 'ClaudeMod')
        )

    def _load_config(self) -> dict:
        """Load and validate configuration"""
        try:
            # Load base config
            config = self._load_yaml_config()
            
            # Validate config values
            self._validate_config(config)
            
            # Create backup
            self._backup_config(config)
            
            return config
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)

    def _load_yaml_config(self) -> dict:
        """Load YAML configuration with fallback to defaults"""
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH) as f:
                    user_config = yaml.safe_load(f)
                    return {**DEFAULT_CONFIG, **user_config}
            
            # If config doesn't exist, create it with defaults
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return DEFAULT_CONFIG

    def _validate_config(self, config: dict) -> None:
        """Validate configuration values"""
        required_sections = ['bot', 'claude', 'database', 'limits']
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required config section: {section}")

        validators = {
            'claude.model': lambda x: isinstance(x, str) and x.startswith('claude-3-'),
            'claude.max_tokens': lambda x: isinstance(x, int) and 1000 <= x <= 4096,
            'claude.temperature': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
            'claude.max_memory': lambda x: isinstance(x, int) and 5 <= x <= 50,
            'limits.rate_limit': lambda x: isinstance(x, int) and 1 <= x <= 100,
        }

        for path, validator in validators.items():
            section, key = path.split('.')
            if not validator(config[section][key]):
                raise ConfigurationError(f"Invalid value for {path}: {config[section][key]}")

        # Add validation for new config sections
        if 'database' not in config:
            raise ConfigurationError("Missing database configuration")
        
        # Validate database config
        db_config = config['database']
        required_db_fields = ['path', 'backup_interval', 'max_attachment_size']
        for field in required_db_fields:
            if field not in db_config:
                raise ConfigurationError(f"Missing required database config: {field}")

    def _backup_config(self, config: dict) -> None:
        """Create a backup of the current config"""
        backup_dir = Path(self.env.backup_path)
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"config-{timestamp}.yaml"
        
        try:
            with open(backup_path, 'w') as f:
                yaml.dump(config, f)
            
            # Keep only last 5 backups
            backups = sorted(backup_dir.glob("config-*.yaml"))
            for old_backup in backups[:-5]:
                old_backup.unlink()
                
        except Exception as e:
            logger.warning(f"Failed to create config backup: {e}")

    def save_config(self, config: dict = None) -> None:
        try:
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config or self.config, f)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def setup_commands(self) -> None:
        @self.bot.command(name='set_prompt')
        async def set_prompt(ctx, *, prompt: str):
            """Set a custom system prompt for Claude"""
            try:
                self.config['system_prompt'] = prompt
                self.save_config()
                await ctx.send("System prompt updated successfully!")
            except Exception as e:
                await ctx.send(f"Error updating prompt: {e}")

        @self.bot.command(name='get_prompt')
        async def get_prompt(ctx):
            """View the current system prompt"""
            embed = Embed(title="Current System Prompt", 
                        description=self.config['system_prompt'],
                        color=0xda7756)
            await ctx.send(embed=embed)

        @self.bot.command(name='set_temperature')
        async def set_temperature(ctx, temp: float):
            """Set Claude's temperature (0.0-1.0)"""
            if 0 <= temp <= 1:
                self.config['temperature'] = temp
                self.save_config()
                await ctx.send(f"Temperature set to {temp}")
            else:
                await ctx.send("Temperature must be between 0 and 1")

        @self.bot.command(name='set_max_memory')
        async def set_max_memory(ctx, size: int):
            """Set maximum conversation history size"""
            if 5 <= size <= 50:  # reasonable limits
                self.config['max_memory'] = size
                self.save_config()
                await ctx.send(f"Maximum conversation history set to {size} messages")
            else:
                await ctx.send("Memory size must be between 5 and 50 messages")

        @self.bot.command(name='set_max_tokens')
        async def set_max_tokens(ctx, tokens: int):
            """Set maximum response tokens"""
            if 1000 <= tokens <= 4096:
                self.config['max_tokens'] = tokens
                self.save_config()
                await ctx.send(f"Maximum tokens set to {tokens}")
            else:
                await ctx.send("Token limit must be between 1000 and 4096")

        @self.bot.command(name='config')
        async def show_config(ctx):
            """Show current configuration"""
            embed = Embed(title="Current Configuration", color=0xda7756)
            for key, value in self.config.items():
                if key != 'system_prompt':  # Handle separately due to length
                    embed.add_field(name=key, value=str(value), inline=True)
            await ctx.send(embed=embed)

        @self.bot.command(name='export_history')
        async def export_history(ctx):
            """Export conversation history as JSON"""
            user_id = str(ctx.author.id)
            history = await self.storage.get_convo(user_id)
            
            with BytesIO(json.dumps(history, indent=2).encode()) as bio:
                await ctx.send("Here's your conversation history:", 
                             file=discord.File(bio, 'history.json'))

        @self.bot.command(name='help')
        async def show_help(ctx):
            """Show detailed help information"""
            embed = Embed(title="ClaudeCord Help", color=0xda7756)
            embed.add_field(name="Basic Usage", 
                          value="Mention the bot with your message to chat", 
                          inline=False)
            # Add other command descriptions
            await ctx.send(embed=embed)

        @self.bot.command(name='set_system_prompt')
        @self.permissions.requires_permission(PermissionLevel.ADMIN)
        async def set_system_prompt(ctx, *, prompt: str):
            """Set the system prompt (Admin only)"""
            try:
                self.config['claude']['system_prompt'] = prompt
                self.save_config()
                await ctx.send("System prompt updated successfully!")
            except Exception as e:
                await ctx.send(f"Error updating system prompt: {e}")

        @self.bot.command(name='ban_user')
        @self.permissions.requires_permission(PermissionLevel.MODERATOR)
        async def ban_user(ctx, user_id: str):
            """Ban a user from using the bot (Mod only)"""
            try:
                # Implement ban logic
                await ctx.send(f"User {user_id} has been banned from using the bot.")
            except Exception as e:
                await ctx.send(f"Error banning user: {e}")

        @commands.command(name='style')
        async def set_style(self, ctx, style: str):
            """Set conversation style"""
            styles = {
                'formal': "maintaining a professional and formal tone",
                'casual': "keeping responses casual and friendly",
                'creative': "being more creative and expressive",
                'technical': "focusing on technical accuracy and detail",
                'simple': "using simple and clear language"
            }
            
            if style.lower() not in styles:
                await ctx.send(f"Available styles: {', '.join(styles.keys())}")
                return
                
            user_id = str(ctx.author.id)
            prompt = f"You are a helpful AI assistant, {styles[style.lower()]}"
            # Store user-specific prompt
            await self.storage.update_user_settings(user_id, {'style': style, 'prompt': prompt})
            await ctx.send(f"Conversation style set to: {style}")

        @commands.command(name='language')
        async def set_language(self, ctx, lang: str):
            """Set preferred response language"""
            user_id = str(ctx.author.id)
            await self.storage.update_user_settings(user_id, {'language': lang})
            await ctx.send(f"Response language set to: {lang}")
            
        @self.bot.command(name='voice')
        async def set_voice(self, ctx, voice: str):
            """Set AI personality/voice"""
            voices = {
                'professional': "an experienced professional",
                'friendly': "a friendly and approachable assistant",
                'expert': "a subject matter expert",
                'teacher': "a patient teacher",
                'mentor': "a helpful mentor"
            }
            
            if voice.lower() not in voices:
                await ctx.send(f"Available voices: {', '.join(voices.keys())}")
                return
                
            user_id = str(ctx.author.id)
            await self.storage.update_user_settings(user_id, {'voice': voice})
            await ctx.send(f"AI voice set to: {voice}")

        @self.bot.command(name='theme')
        async def set_theme(ctx, theme_name: str):
            """Set your preferred theme"""
            try:
                theme = Theme[theme_name.upper()]
                self.theme_manager.set_user_theme(str(ctx.author.id), theme)
                
                # Create preview embed with new theme
                colors = self.theme_manager.get_user_theme(str(ctx.author.id))
                embed = Embed(
                    title="Theme Updated",
                    description=f"Your theme has been set to: {theme_name}",
                    color=colors.primary
                )
                embed.add_field(name="Primary Color", value=hex(colors.primary))
                embed.add_field(name="Secondary Color", value=hex(colors.secondary))
                
                await ctx.send(embed=embed)
            except KeyError:
                available_themes = ", ".join(t.value for t in Theme)
                await ctx.send(f"Invalid theme. Available themes: {available_themes}")

        @self.bot.command(name='format')
        async def set_format(ctx, style: str):
            """Set your preferred formatting style"""
            try:
                format_style = FormatStyle[style.upper()]
                self.embed_manager.set_user_preference(
                    str(ctx.author.id), 
                    {'style': format_style}
                )
                
                # Show preview
                preview = (
                    "# Heading\n"
                    "Some regular text\n"
                    "```python\ndef hello(): print('world')\n```\n"
                    "- Bullet point\n"
                    "> Quote block\n"
                    "| Column 1 | Column 2 |\n"
                    "| -------- | -------- |\n"
                    "| Data 1   | Data 2   |"
                )
                
                embeds = self.embed_manager.create_response_embed(
                    preview,
                    "Format Preview",
                    user_id=str(ctx.author.id)
                )
                await ctx.send(embeds=embeds)
            except KeyError:
                available_styles = ", ".join(s.value for s in FormatStyle)
                await ctx.send(f"Invalid style. Available styles: {available_styles}")

        @self.bot.command(name='settings')
        async def show_settings(ctx):
            """Show and modify your settings"""
            user_id = str(ctx.author.id)
            settings = await self.storage.get_user_settings(user_id)
            
            view = SettingsView(settings)
            embed = Embed(title="User Settings", color=self.theme_manager.get_user_theme(user_id).primary)
            
            # Add current settings
            for key, value in settings.items():
                embed.add_field(name=key, value=str(value), inline=True)
                
            message = await ctx.send(embed=embed, view=view)
            
            # Wait for interaction
            try:
                await view.wait()
                if view.value:
                    # Update settings
                    new_settings = {**settings, **view.value}
                    await self.storage.update_user_settings(user_id, new_settings)
                    await message.edit(content="Settings updated!", embed=embed, view=None)
            except TimeoutError:
                await message.edit(content="Settings menu expired.", view=None)

        @self.bot.command(name='context')
        async def manage_context(ctx, action: str = "show", *, content: str = None):
            """Manage conversation context"""
            user_id = str(ctx.author.id)
            context = await self.context_manager.get_context(user_id)
            
            if action == "show":
                # Show current context summary
                embed = Embed(
                    title="Current Context",
                    description=context.metadata.get("summary", "No context summary available"),
                    color=self.theme_manager.get_user_theme(user_id).primary
                )
                embed.add_field(
                    name="Messages", 
                    value=str(context.metadata["total_messages"])
                )
                embed.add_field(
                    name="Last Updated",
                    value=context.metadata["last_updated"]
                )
                await ctx.send(embed=embed)
                
            elif action == "clear":
                view = ConfirmationView()
                msg = await ctx.send("Are you sure you want to clear your context?", view=view)
                
                await view.wait()
                if view.value:
                    context.clear_context()
                    await self.context_manager.save_context(user_id)
                    await msg.edit(content="Context cleared!", view=None)
                else:
                    await msg.edit(content="Context clear cancelled.", view=None)
                    
            elif action == "add":
                if content:
                    context.add_message("user", content, {"type": "context"})
                    await self.context_manager.save_context(user_id)
                    await ctx.send("Context updated!")
                else:
                    await ctx.send("Please provide context to add!")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_claude_response(self, user_id: str, content: List[Dict[str, any]]) -> str:
        # Add validation for content structure
        if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
            raise ValueError("Invalid content format")
        
        # Add timeout handling
        async with asyncio.timeout(30):  # 30 second timeout
            try:
                conversation = await self.storage.get_convo(user_id)
                
                # Implement smarter conversation pruning
                total_tokens = 0
                pruned_conversation = []
                
                for msg in reversed(conversation):
                    estimated_tokens = len(str(msg).encode('utf-8')) // 4  # rough estimate
                    if total_tokens + estimated_tokens <= self.config['max_tokens'] * 0.75:  # Leave room for response
                        pruned_conversation.insert(0, msg)
                        total_tokens += estimated_tokens
                    else:
                        break
                
                # process attachment references
                for item in content:
                    if item['type'] == 'image' and item['source']['type'] == 'attachment_ref':
                        attachment_id = item['source']['attachment_id']
                        filename, content = await self.storage.get_attachment(attachment_id)
                        item['source'] = {"type": "base64", "media_type": "image/png", "data": content}
                
                # add the new content to the conversation
                conversation.append({"role": "user", "content": content})
                
                # trim the conversation if it's too long
                if len(conversation) > self.config['max_memory']:
                    conversation = conversation[-(self.config['max_memory'] - (self.config['max_memory'] % 2)):]
                
                msg = await self.claude_client.messages.create(
                    model=self.config['model'],
                    max_tokens=self.config['max_tokens'],
                    temperature=self.config['temperature'],
                    system=self.config['system_prompt'],
                    messages=conversation
                )
                
                assistant_response = msg.content[0].text
                
                # Add error handling for response validation
                if not assistant_response or len(assistant_response) < 1:
                    raise ValueError("Empty response from Claude")
                
                # add Claude's response to the conversation
                conversation.append({"role": "assistant", "content": assistant_response})
                
                # update the entire conversation in the database
                await self.storage.update_convo(user_id, conversation)
                
                logger.debug(f"Processed message for user {user_id}")
                logger.debug(f"Conversation history: {conversation}")
                return assistant_response
            except asyncio.TimeoutError:
                logger.error("Claude API request timed out")
                raise
            except Exception as e:
                logger.error(f"Error in get_claude_response: {e}")
                if "rate limit" in str(e).lower():
                    raise RateLimitError("Rate limit exceeded")
                raise

    async def validate_content(self, content: List[Dict[str, any]]) -> bool:
        """Validate content before sending to Claude"""
        total_size = 0
        for item in content:
            if item['type'] == 'text':
                total_size += len(item['text'].encode('utf-8'))
            elif item['type'] == 'image':
                if item['source']['type'] == 'base64':
                    total_size += len(item['source']['data'])
                    
        return total_size <= 100_000_000  # 100MB limit

    async def send_msg(self, msg: Message, content: List[Dict[str, any]]) -> None:
        """Enhanced message sending with monitoring"""
        start_time = time.time()
        user_id = str(msg.author.id)
        
        try:
            # Existing message sending code...
            self.monitoring.record_message(user_id)
            response_time = time.time() - start_time
            self.monitoring.record_response_time(response_time)
        except Exception as e:
            self.monitoring.record_error()
            logger.error(f"Error sending message: {e}")
            raise

    async def _process_message(self, message_data: Dict[str, Any]) -> None:
        """Process a message from the queue"""
        msg = message_data['msg']
        content = message_data['content']

        if not content:
            logger.warning('Content was empty.')
            return

        if not await self.validate_content(content):
            await msg.channel.send("Content size exceeds limits. Please try with less data.")
            return

        try:
            thinking_msg = await msg.channel.send("Thinking 🤔...")
            claude_response: str = await self.get_claude_response(str(msg.author.id), content)
            await thinking_msg.delete()
            
            # split the response into chunks of 2000 characters or less
            chunks = [claude_response[i:i+2000] for i in range(0, len(claude_response), 2000)]

            for chunk in chunks:
                embed = Embed(description=chunk, color=0xda7756)
                await msg.channel.send(embed=embed)
             
            logger.debug(f"Sent response to user {msg.author.id}")

        except Exception as e:
            logger.error(f"An error occurred in send_msg: {e}", exc_info=True)
            await msg.channel.send("I'm sorry, I encountered an error while processing your request.")

    async def _rotate_status(self):
        """Rotate through different status messages"""
        while True:
            message, activity_type = self.status_messages[self.status_index]
            activity = Activity(type=activity_type, name=message)
            await self.bot.change_presence(activity=activity, status=Status.online)
            
            self.status_index = (self.status_index + 1) % len(self.status_messages)
            await asyncio.sleep(60)  # Change status every minute

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        logger.info(f'{self.bot.user} is now running...')
        await self.storage.init()
        
        # Start status rotation
        self._status_task = asyncio.create_task(self._rotate_status())
        
        # Add cool startup message with ASCII art
        ascii_art = """
        ╔═══════════════════════════════════════╗
        ║          ClaudeCord is Online!        ║
        ║    Powered by Claude 3.5 Sonnet       ║
        ║    Ready to assist and chat! 🤖✨     ║
        ╚═══════════════════════════════════════╝
        """
        logger.info(ascii_art)
        
        # check if PyNaCl is installed
        try:
            import nacl
            logger.info("PyNaCl is installed. Voice support is available.")
        except ImportError:
            logger.warning("PyNaCl is not installed. Voice will NOT be supported.")

    @commands.Cog.listener()
    async def on_message(self, msg: Message) -> None:
        if msg.author == self.bot.user:
            return

        if self.bot.user.mentioned_in(msg):
            content = []
            
            # process text
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            

            reading_msg = await msg.channel.send("reading your attachments 🔎...")

            # process attachments
            for attachment in msg.attachments:
                attachment_content = await process_file(attachment, str(msg.author.id), self.storage)
                content.extend(attachment_content)
            
            await reading_msg.delete()

            if content:
                await self.send_msg(msg, content)
            else:
                await msg.channel.send("Please provide some text, images, or files for me to analyze.")
        
        await self.bot.process_commands(msg)

    @bot.command(name='delete_history')
    async def delete_history(self, ctx):
        user_id = str(ctx.author.id)
        confirm_msg = await ctx.send("Are you sure you want to delete your entire conversation history? This action cannot be undone. Reply with 'y' to confirm.")
        
        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == 'y'
        
        try:
            await self.bot.wait_for('message', check=check, timeout=30.0)
        except asyncio.TimeoutError:
            await confirm_msg.edit(content="Deletion cancelled. You did not confirm in time.")
        else:
            await self.storage.delete_user_convo(user_id)
            await ctx.send("Your conversation history has been deleted.")

    async def start(self):
        try:
            await self.storage.init()
            await self.monitoring.start_monitoring()
            await self.bot.start(DISCORD_TOK)
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            raise
            
    async def close(self):
        """Cleanup bot resources"""
        try:
            # Cancel status rotation task
            if self._status_task:
                self._status_task.cancel()
                
            # Cancel all pending tasks
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
                
            # Add database connection pool cleanup
            if hasattr(self, 'db_pool'):
                await self.db_pool.close()
                
            # Add explicit cleanup of message queue
            if hasattr(self, 'message_queue'):
                await self.message_queue.stop()
                
            # Stop monitoring
            await self.monitoring.stop_monitoring()
            
            # Save all contexts
            await self.context_manager.stop()
            
            # Close database connections
            await self.storage.close()
            
            # Close Discord connection
            await self.bot.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
        finally:
            # Ensure critical resources are cleaned up
            for task in asyncio.all_tasks():
                task.cancel()

    @commands.command(name='health')
    @commands.has_permissions(administrator=True)
    async def health_check(self, ctx):
        """Show bot health status"""
        health_status = self.monitoring.get_health_status()
        
        embed = Embed(
            title="Bot Health Status",
            color=0x2ecc71 if health_status["status"] == "healthy" else 0xe74c3c
        )
        
        # Add health check results
        for check, status in health_status["checks"].items():
            embed.add_field(
                name=f"{check.title()} Status",
                value="✅" if status else "❌",
                inline=True
            )
            
        # Add metrics
        metrics = health_status["metrics"]
        embed.add_field(name="Uptime", value=f"{metrics['uptime']:.2f}s", inline=True)
        embed.add_field(name="Total Messages", value=metrics['total_messages'], inline=True)
        embed.add_field(name="Error Rate", value=f"{metrics['error_rate']:.2f}%", inline=True)
        embed.add_field(name="Avg Response Time", value=f"{metrics['avg_response_time']:.2f}s", inline=True)
        embed.add_field(name="Active Users", value=metrics['active_users'], inline=True)
        embed.add_field(name="Memory Usage", value=f"{metrics['memory_usage']:.1f}%", inline=True)
        
        await ctx.send(embed=embed)

    async def handle_error(self, ctx, error):
        """Global error handler"""
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("You don't have permission to use this command.")
        elif isinstance(error, commands.CommandNotFound):
            await ctx.send("Command not found. Use `>help` to see available commands.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"Missing required argument: {error.param}")
        else:
            logger.error(f"Unhandled error: {error}")
            await ctx.send("An error occurred while processing your command.")

    async def cleanup_resources(self):
        """Cleanup all bot resources"""
        try:
            # Add database connection pool cleanup
            if hasattr(self, 'db_pool'):
                await self.db_pool.close()
                
            # Add explicit cleanup of message queue
            if hasattr(self, 'message_queue'):
                await self.message_queue.stop()
                
            # Stop monitoring
            await self.monitoring.stop_monitoring()
            
            # Save all contexts
            await self.context_manager.stop()
            
            # Close database connections
            await self.storage.close()
            
            # Close Discord connection
            await self.bot.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            # Cancel all pending tasks
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()

    async def process_message(self, message: discord.Message):
        # Sanitize user input
        content = self.sanitizer.clean_content(message.content)

if __name__ == '__main__':
    bot = ClaudeCordBot()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
