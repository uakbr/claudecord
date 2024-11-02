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
from datetime import datetime
from dateutil import parser
from collections import defaultdict
import re
from datetime import timedelta

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
from openai import AsyncOpenAI
import httpx
from typing import AsyncGenerator, Tuple
from async_timeout import timeout

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

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

class RateLimitError(Exception):
    """Raised when rate limits are exceeded"""
    pass

class RateLimiter:
    def __init__(self):
        self.command_limits = defaultdict(lambda: defaultdict(list))
        self.message_limits = defaultdict(list)
        self.blocked_users = set()
        self.suspicious_patterns = set()
        
    def check_rate_limit(self, user_id: str, command: str = None) -> bool:
        now = datetime.now()
        if command:
            # Command-specific limits
            timeframe = timedelta(seconds=30)
            user_commands = self.command_limits[user_id][command]
            user_commands = [t for t in user_commands if now - t < timeframe]
            self.command_limits[user_id][command] = user_commands
            
            if len(user_commands) >= 5:  # Max 5 same commands per 30s
                return False
            self.command_limits[user_id][command].append(now)
        else:
            # General message limits
            timeframe = timedelta(minutes=5)
            user_messages = [t for t in self.message_limits[user_id] if now - t < timeframe]
            self.message_limits[user_id] = user_messages
            
            if len(user_messages) >= 30:  # Max 30 messages per 5min
                return False
            self.message_limits[user_id].append(now)
            
        return True

class SecurityManager:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.spam_detection = defaultdict(list)
        self.message_history = defaultdict(list)
        self.suspicious_users = set()
        
        # Regex patterns for potentially malicious content
        self.dangerous_patterns = [
            r"(?i)(exec|eval|system|os\.|subprocess|import os|import subprocess)",
            r"(?i)(rm -rf|format|del|remove)",
            r"(?:[a-zA-Z0-9+/]{4}){30,}={0,3}",  # Long base64 strings
            r"(?i)(http|ftp|ws)s?://\S+",  # URLs
            r"@everyone|@here",
            r"discord\.gg/\S+",  # Discord invites
        ]
        
    def is_suspicious_content(self, content: str) -> bool:
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content):
                return True
                
        # Check for spam-like content
        if len(content) > 1000:  # Long messages
            return True
            
        if content.count('\n') > 20:  # Too many newlines
            return True
            
        return False
        
    def check_message_similarity(self, user_id: str, content: str) -> bool:
        """Check if user is spamming similar messages"""
        now = datetime.now()
        timeframe = timedelta(minutes=5)
        
        # Update message history
        self.message_history[user_id] = [
            (t, m) for t, m in self.message_history[user_id] 
            if now - t < timeframe
        ]
        self.message_history[user_id].append((now, content))
        
        # Check for repeated similar messages
        recent_messages = self.message_history[user_id]
        if len(recent_messages) >= 3:
            similar_count = sum(1 for _, m in recent_messages 
                              if self._similarity_score(m, content) > 0.8)
            if similar_count >= 3:
                return True
        return False
        
    def _similarity_score(self, s1: str, s2: str) -> float:
        """Simple similarity check"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()

# Add to imports and constants section
SUPER_ADMIN_ID: Final[str] = "YOUR_DISCORD_ID_HERE"  # Replace with your Discord ID
OPENAI_KEY: Final[str] = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL: Final[str] = "gpt-4-turbo-preview"

@dataclass
class ComparisonMetrics:
    """Metrics for comparing LLM responses"""
    token_count: int
    response_time: float
    similarity_score: float
    sentiment_score: float
    complexity_score: float
    
class EnhancedMultiProviderStreamer:
    """Enhanced version of MultiProviderStreamer with advanced features"""
    
    def __init__(self, anthropic_client: AsyncAnthropic, openai_client: AsyncOpenAI):
        self.anthropic = anthropic_client
        self.openai = openai_client
        self.active_streams: Dict[str, Set[str]] = defaultdict(set)
        self.response_cache: Dict[str, Dict[str, str]] = {}
        self.metrics_history: Dict[str, List[ComparisonMetrics]] = defaultdict(list)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    async def stream_both(self, 
                         prompt: str, 
                         user_id: str,
                         max_tokens: int = 4096,
                         analysis: bool = True,
                         style: Literal["conversational", "analytical", "creative"] = "conversational") -> AsyncGenerator[Tuple[str, str, Optional[Dict]], None]:
        """
        Enhanced streaming with real-time analysis
        
        Args:
            prompt: Input prompt
            user_id: User ID
            max_tokens: Max response tokens
            analysis: Enable real-time analysis
            style: Response style preference
        """
        start_time = time.time()
        self.active_streams[user_id] = {"anthropic", "openai"}
        
        # Adapt prompt based on style
        style_prompts = {
            "conversational": "Respond in a natural, conversational manner: ",
            "analytical": "Provide a detailed analytical response: ",
            "creative": "Respond creatively and imaginatively: "
        }
        enhanced_prompt = f"{style_prompts[style]}{prompt}"
        
        try:
            async with timeout(30):
                claude_stream = self.anthropic.messages.stream(
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    model=MODEL_NAME,
                    max_tokens=max_tokens
                )
                
                openai_stream = self.openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    max_tokens=max_tokens,
                    stream=True
                )
                
                # Track responses for analysis
                claude_response = ""
                gpt4_response = ""
                
                async def process_claude():
                    nonlocal claude_response
                    try:
                        async for chunk in claude_stream:
                            if "anthropic" in self.active_streams[user_id]:
                                token = chunk.delta.text or ""
                                claude_response += token
                                analysis_data = await self._analyze_responses(
                                    claude_response, gpt4_response
                                ) if analysis else None
                                yield ("Claude", token, analysis_data)
                    except Exception as e:
                        logger.error(f"Claude stream error: {e}")
                    finally:
                        self.active_streams[user_id].discard("anthropic")
                        
                async def process_openai():
                    nonlocal gpt4_response
                    try:
                        async for chunk in openai_stream:
                            if "openai" in self.active_streams[user_id]:
                                token = chunk.choices[0].delta.content or ""
                                gpt4_response += token
                                analysis_data = await self._analyze_responses(
                                    claude_response, gpt4_response
                                ) if analysis else None
                                yield ("GPT-4", token, analysis_data)
                    except Exception as e:
                        logger.error(f"OpenAI stream error: {e}")
                    finally:
                        self.active_streams[user_id].discard("openai")
                
                # Process streams with analysis
                async for response in asyncio.gather(
                    process_claude(), 
                    process_openai()
                ):
                    yield response
                    
                # Store final responses and metrics
                end_time = time.time()
                if claude_response and gpt4_response:
                    self.response_cache[user_id] = {
                        "claude": claude_response,
                        "gpt4": gpt4_response
                    }
                    
                    metrics = ComparisonMetrics(
                        token_count=len(self.tokenizer.encode(claude_response)),
                        response_time=end_time - start_time,
                        similarity_score=self._calculate_similarity(claude_response, gpt4_response),
                        sentiment_score=await self._analyze_sentiment(claude_response, gpt4_response),
                        complexity_score=self._calculate_complexity(claude_response, gpt4_response)
                    )
                    self.metrics_history[user_id].append(metrics)
                    
        except asyncio.TimeoutError:
            logger.error("Streaming timed out")
        finally:
            self.active_streams.pop(user_id, None)
            
    async def _analyze_responses(self, claude_resp: str, gpt4_resp: str) -> Dict:
        """Real-time response analysis"""
        if not claude_resp or not gpt4_resp:
            return None
            
        return {
            "similarity": self._calculate_similarity(claude_resp, gpt4_resp),
            "length_diff": len(gpt4_resp) - len(claude_resp),
            "claude_tokens": len(self.tokenizer.encode(claude_resp)),
            "gpt4_tokens": len(self.tokenizer.encode(gpt4_resp)),
            "complexity_diff": self._calculate_complexity(claude_resp, gpt4_resp)
        }
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score"""
        return SequenceMatcher(None, text1, text2).ratio()
        
    async def _analyze_sentiment(self, text1: str, text2: str) -> float:
        """Analyze sentiment consistency between responses"""
        # Simplified sentiment analysis
        positive_words = {"good", "great", "excellent", "positive", "agree"}
        negative_words = {"bad", "poor", "negative", "disagree"}
        
        def get_sentiment(text):
            text = text.lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            return (pos_count - neg_count) / (pos_count + neg_count + 1)
            
        sent1 = get_sentiment(text1)
        sent2 = get_sentiment(text2)
        return 1 - abs(sent1 - sent2)  # Similarity score
        
    def _calculate_complexity(self, text1: str, text2: str) -> float:
        """Calculate response complexity difference"""
        def complexity_score(text):
            sentences = text.split(". ")
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            unique_words = len(set(text.lower().split()))
            return (avg_sentence_length * 0.5) + (unique_words * 0.5)
            
        return abs(complexity_score(text1) - complexity_score(text2))

    async def get_comparison_summary(self, user_id: str) -> Dict:
        """Get detailed comparison summary"""
        if user_id not in self.response_cache:
            return None
            
        responses = self.response_cache[user_id]
        metrics = self.metrics_history[user_id][-1] if self.metrics_history[user_id] else None
        
        return {
            "responses": responses,
            "metrics": metrics,
            "analysis": {
                "token_efficiency": len(self.tokenizer.encode(responses["claude"])) / 
                                  len(self.tokenizer.encode(responses["gpt4"])),
                "response_consistency": self._calculate_similarity(
                    responses["claude"], responses["gpt4"]
                ),
                "complexity_comparison": self._calculate_complexity(
                    responses["claude"], responses["gpt4"]
                )
            }
        }

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
            ("with neural networks ", ActivityType.playing),
            ("your conversations 💭", ActivityType.watching),
            ("to your requests 👂", ActivityType.listening),
            ("Claude 3.5 Sonnet 🤖", ActivityType.competing),
            ("Need help? Mention me! ✨", ActivityType.custom),
        ]
        self._status_task = None
        self.security = SecurityManager()
        self.super_admin = SUPER_ADMIN_ID
        self.openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
        self.streamer = EnhancedMultiProviderStreamer(self.claude_client, self.openai_client)
        # Add task tracking
        self._tasks = set()

    def is_super_admin(self, user_id: str) -> bool:
        """Check if user is the super admin"""
        return user_id == self.super_admin
        
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
        async def show_help(ctx, query: Optional[str] = None):
            """Show detailed help information"""
            help_system = EnhancedHelpCommand(self.bot)
            
            if not query:
                # Show main help overview
                embed = await help_system.create_overview_embed()
                view = PaginationView()
                message = await ctx.send(embed=embed, view=view)
                
                # Handle pagination
                current_page = 0
                embeds = [embed]
                for category in help_system.categories:
                    embeds.append(await help_system.create_category_embed(category))
                    
                async def update_page(interaction, direction):
                    nonlocal current_page
                    if direction == "next":
                        current_page = (current_page + 1) % len(embeds)
                    else:
                        current_page = (current_page - 1) % len(embeds)
                    await message.edit(embed=embeds[current_page])
                    await interaction.response.defer()
                    
                view.next_page.callback = lambda i: update_page(i, "next")
                view.prev_page.callback = lambda i: update_page(i, "prev")
                
            else:
                # Search for specific category or command
                query = query.lower()
                
                # Check categories
                for category in help_system.categories:
                    if query == category.name.lower():
                        embed = await help_system.create_category_embed(category)
                        await ctx.send(embed=embed)
                        return
                        
                # Check commands
                for category in help_system.categories:
                    for cmd in category.commands:
                        if query == cmd['name'].lower():
                            embed = Embed(
                                title=f"Command: {cmd['name']}",
                                description=cmd['description'],
                                color=0xda7756
                            )
                            embed.add_field(
                                name="Usage",
                                value=f"`{cmd['usage']}`",
                                inline=False
                            )
                            embed.add_field(
                                name="Examples",
                                value="\n".join(f"`{ex}`" for ex in cmd['examples']),
                                inline=False
                            )
                            await ctx.send(embed=embed)
                            return
                            
                await ctx.send(f"No help found for '{query}'. Use `!help` to see all categories and commands.")

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

        @commands.command(name="serverstats")
        async def server_stats(self, ctx):
            """Display server statistics"""
            guild = ctx.guild
            embed = Embed(title=f"📊 {guild.name} Statistics", color=0xda7756)
            
            # Member stats
            total_members = len(guild.members)
            online_members = len([m for m in guild.members if m.status != Status.offline])
            bots = len([m for m in guild.members if m.bot])
            
            # Channel stats
            text_channels = len(guild.text_channels)
            voice_channels = len(guild.voice_channels)
            categories = len(guild.categories)
            
            # Server info
            embed.add_field(name="👥 Members", value=f"Total: {total_members}\nOnline: {online_members}\nBots: {bots}")
            embed.add_field(name="💬 Channels", value=f"Text: {text_channels}\nVoice: {voice_channels}\nCategories: {categories}")
            embed.add_field(name="🎭 Roles", value=str(len(guild.roles)))
            embed.add_field(name="📅 Created", value=guild.created_at.strftime("%Y-%m-%d"))
            embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
            
            await ctx.send(embed=embed)

        @commands.command(name="profile")
        async def show_profile(self, ctx, member: discord.Member = None):
            """Show user profile with stats and level"""
            member = member or ctx.author
            user_id = str(member.id)
            
            # Get user stats from database
            stats = await self.storage.get_user_stats(user_id)
            level = calculate_level(stats['xp'])
            
            embed = Embed(title=f"{member.display_name}'s Profile", color=0xda7756)
            embed.set_thumbnail(url=member.display_avatar.url)
            embed.add_field(name="Level", value=str(level))
            embed.add_field(name="XP", value=f"{stats['xp']}/{calculate_next_level_xp(level)}")
            embed.add_field(name="Messages", value=str(stats['messages']))
            embed.add_field(name="Commands Used", value=str(stats['commands']))
            
            await ctx.send(embed=embed)

        @commands.command(name="remind")
        async def set_reminder(self, ctx, time: str, *, reminder: str):
            """Set a reminder"""
            # Parse time string (e.g., "2h30m", "1d", etc.)
            duration = parse_time_string(time)
            if not duration:
                await ctx.send("Invalid time format! Use format like '2h30m' or '1d'")
                return
                
            reminder_time = datetime.now() + duration
            await self.storage.add_reminder(
                user_id=str(ctx.author.id),
                channel_id=str(ctx.channel.id),
                reminder=reminder,
                remind_at=reminder_time
            )
            
            await ctx.send(f"I'll remind you about '{reminder}' in {time}!")

        @commands.command(name="addcmd")
        @commands.has_permissions(manage_guild=True)
        async def add_custom_command(self, ctx, cmd_name: str, *, response: str):
            """Add a custom command"""
            await self.storage.add_custom_command(
                guild_id=str(ctx.guild.id),
                name=cmd_name,
                response=response
            )
            await ctx.send(f"Added custom command `{cmd_name}`!")

        @commands.command(name="poll")
        async def create_poll(self, ctx, question: str, *options):
            """Create a poll with reactions"""
            if len(options) < 2:
                await ctx.send("Please provide at least 2 options!")
                return
                
            # Limit options to 10 (0-9 reactions)
            options = options[:10]
            
            # Create poll embed
            embed = Embed(title="📊 Poll", description=question, color=0xda7756)
            
            # Add options with numbers
            for i, opt in enumerate(options):
                embed.add_field(name=f"Option {i+1}", value=opt, inline=False)
            
            poll_msg = await ctx.send(embed=embed)
            
            # Add number reactions
            for i in range(len(options)):
                await poll_msg.add_reaction(f"{i+1}\u20e3")

        # Add new super admin commands
        @self.bot.command(name='sudo')
        async def sudo_command(ctx, *, command: str):
            """Execute any command with super admin privileges"""
            if not self.is_super_admin(str(ctx.author.id)):
                await ctx.send("This command is restricted to super admin.")
                return
                
            try:
                # Parse and execute the command
                cmd_parts = command.split()
                cmd_name = cmd_parts[0]
                cmd_args = cmd_parts[1:]
                
                cmd = self.bot.get_command(cmd_name)
                if cmd:
                    ctx.command = cmd
                    await ctx.invoke(cmd, *cmd_args)
                else:
                    await ctx.send(f"Command '{cmd_name}' not found.")
            except Exception as e:
                await ctx.send(f"Error executing command: {e}")

        @self.bot.command(name='system')
        async def system_command(ctx, *, action: str):
            """Perform system-level actions (super admin only)"""
            if not self.is_super_admin(str(ctx.author.id)):
                await ctx.send("This command is restricted to super admin.")
                return
                
            try:
                if action == "restart":
                    await ctx.send("Restarting bot...")
                    await self.bot.close()
                    os.execv(sys.executable, ['python'] + sys.argv)
                    
                elif action == "shutdown":
                    await ctx.send("Shutting down bot...")
                    await self.bot.close()
                    sys.exit(0)
                    
                elif action.startswith("config"):
                    # Direct config manipulation
                    _, key, value = action.split(maxsplit=2)
                    self.config[key] = value
                    self.save_config()
                    await ctx.send(f"Config updated: {key} = {value}")
                    
                elif action == "debug":
                    # Toggle debug mode
                    debug_status = "enabled" if logging.getLogger().level == logging.DEBUG else "disabled"
                    await ctx.send(f"Debug mode is {debug_status}")
                    
                else:
                    await ctx.send("Unknown system action")
                    
            except Exception as e:
                await ctx.send(f"Error executing system action: {e}")

        @self.bot.command(name='override')
        async def override_command(ctx, user_id: str, *, action: str):
            """Override user permissions and restrictions (super admin only)"""
            if not self.is_super_admin(str(ctx.author.id)):
                await ctx.send("This command is restricted to super admin.")
                return
                
            try:
                if action == "unban":
                    self.security.rate_limiter.blocked_users.discard(user_id)
                    await ctx.send(f"User {user_id} has been unbanned.")
                    
                elif action == "reset_limits":
                    self.security.rate_limiter.command_limits.pop(user_id, None)
                    self.security.rate_limiter.message_limits.pop(user_id, None)
                    await ctx.send(f"Rate limits reset for user {user_id}")
                    
                elif action == "clear_history":
                    await self.storage.delete_user_convo(user_id)
                    await ctx.send(f"Conversation history cleared for user {user_id}")
                    
                else:
                    await ctx.send("Unknown override action")
                    
            except Exception as e:
                await ctx.send(f"Error executing override: {e}")

        @self.bot.command(name='compare')
        async def compare_responses(self, ctx, style: str = "conversational", *, prompt: str):
            """Compare responses with real-time analysis"""
            async with ctx.typing():
                # Create embeds for both providers
                claude_embed = Embed(title="Claude's Response", color=0x9932CC)
                gpt4_embed = Embed(title="GPT-4's Response", color=0x32CD32)
                analysis_embed = Embed(title="Real-time Analysis", color=0x4169E1)
                
                claude_response = ""
                gpt4_response = ""
                
                # Send initial embeds
                claude_msg = await ctx.send(embed=claude_embed)
                gpt4_msg = await ctx.send(embed=gpt4_embed)
                analysis_msg = await ctx.send(embed=analysis_embed)
                
                try:
                    async for provider, token, analysis in self.streamer.stream_both(
                        prompt, str(ctx.author.id), style=style
                    ):
                        if provider == "Claude":
                            claude_response += token
                            claude_embed.description = claude_response[:4000]
                            await claude_msg.edit(embed=claude_embed)
                        else:
                            gpt4_response += token
                            gpt4_embed.description = gpt4_response[:4000]
                            await gpt4_msg.edit(embed=gpt4_embed)
                            
                        if analysis:
                            analysis_embed.clear_fields()
                            analysis_embed.add_field(
                                name="Similarity",
                                value=f"{analysis['similarity']:.2%}",
                                inline=True
                            )
                            analysis_embed.add_field(
                                name="Token Usage",
                                value=f"Claude: {analysis['claude_tokens']}\nGPT-4: {analysis['gpt4_tokens']}",
                                inline=True
                            )
                            analysis_embed.add_field(
                                name="Complexity Difference",
                                value=f"{analysis['complexity_diff']:.2f}",
                                inline=True
                            )
                            await analysis_msg.edit(embed=analysis_embed)
                            
                except Exception as e:
                    await ctx.send(f"Error during comparison: {e}")
                    logger.error(f"Comparison error: {e}")
                    
                finally:
                    await self.streamer.stop_streams(str(ctx.author.id))
                    
                # Add final summary
                summary = await self.streamer.get_comparison_summary(str(ctx.author.id))
                if summary:
                    summary_embed = Embed(title="Comparison Summary", color=0xFFD700)
                    summary_embed.add_field(
                        name="Token Efficiency",
                        value=f"{summary['analysis']['token_efficiency']:.2f}",
                        inline=True
                    )
                    summary_embed.add_field(
                        name="Response Consistency",
                        value=f"{summary['analysis']['response_consistency']:.2%}",
                        inline=True
                    )
                    summary_embed.add_field(
                        name="Complexity Comparison",
                        value=f"{summary['analysis']['complexity_comparison']:.2f}",
                        inline=True
                    )
                    await ctx.send(embed=summary_embed)

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

    def create_task(self, coro, name=None) -> asyncio.Task:
        """Create and track a new task"""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _rotate_status(self):
        """Rotate through different status messages"""
        while True:
            try:
                message, activity_type = self.status_messages[self.status_index]
                activity = Activity(type=activity_type, name=message)
                await self.bot.change_presence(activity=activity, status=Status.online)
                
                self.status_index = (self.status_index + 1) % len(self.status_messages)
                await asyncio.sleep(60)  # Change status every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status rotation: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        logger.info(f'{self.bot.user} is now running...')
        await self.storage.init()
        
        # Start status rotation with tracked task
        self._status_task = self.create_task(self._rotate_status(), "status_rotation")
        
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
    async def on_message(self, message: discord.Message):
        """Process and monitor all messages"""
        try:
            # Skip bot messages
            if message.author.bot:
                return

            # Auto-moderation checks
            if message.guild:  # Only check server messages
                settings = await self.storage.get_guild_settings(str(message.guild.id))
                if settings.get('filter_enabled'):
                    content = message.content.lower()
                    if any(word in content for word in settings.get('filtered_words', [])):
                        await message.delete()
                        await message.channel.send(
                            f"{message.author.mention} That word is not allowed here!",
                            delete_after=5
                        )
                        return

            # Process message with security checks
            await self.process_message(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")

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
            if self._status_task and not self._status_task.done():
                self._status_task.cancel()
                try:
                    await self._status_task
                except asyncio.CancelledError:
                    pass

            # Cancel all tracked tasks
            remaining_tasks = list(self._tasks)
            if remaining_tasks:
                logger.info(f"Cancelling {len(remaining_tasks)} remaining tasks")
                for task in remaining_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*remaining_tasks, return_exceptions=True)
            
            # Cleanup resources
            if hasattr(self, 'db_pool'):
                await self.db_pool.close()
            if hasattr(self, 'message_queue'):
                await self.message_queue.stop()
            await self.monitoring.stop_monitoring()
            await self.context_manager.stop()
            await self.storage.close()
            await self.bot.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
        finally:
            # Cancel all pending tasks
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
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
        """Process and validate incoming messages"""
        user_id = str(message.author.id)
        content = message.content
        
        # Skip messages from self
        if message.author.id == self.bot.user.id:
            return
            
        # Check if user is rate limited
        if not self.security.rate_limiter.check_rate_limit(user_id):
            await message.add_reaction('⏳')
            return
            
        # Check for suspicious content
        if self.security.is_suspicious_content(content):
            await message.delete()
            await message.channel.send(
                f"{message.author.mention} Your message was removed for security reasons.",
                delete_after=5
            )
            self.security.suspicious_users.add(user_id)
            return
            
        # Check for spam patterns
        if self.security.check_message_similarity(user_id, content):
            await message.delete()
            await message.channel.send(
                f"{message.author.mention} Please avoid sending similar messages repeatedly.",
                delete_after=5
            )
            return
            
        # Sanitize content before processing
        content = self.sanitizer.clean_content(content)
        
        # Super admin bypass
        if self.is_super_admin(user_id):
            # Process message directly without restrictions
            await self._process_message({'msg': message, 'content': content})
            return
            
        # Process message normally
        await self._process_message({'msg': message, 'content': content})

    async def setup_help_command(self):
        """Setup paginated help command with categories"""
        help_command = commands.DefaultHelpCommand(
            no_category="General",
            paginator=commands.Paginator(prefix=None, suffix=None),
            sort_commands=True,
            dm_help=True,
            commands_heading="Commands",
            aliases_heading="Aliases"
        )
        self.bot.help_command = help_command

    @commands.Cog.listener()
    async def on_member_join(self, member):
        """Welcome new members with customizable message"""
        welcome_channel = member.guild.system_channel
        if welcome_channel:
            embed = Embed(
                title="Welcome! 👋",
                description=f"Welcome to {member.guild.name}, {member.mention}! \n"
                           f"We're now {len(member.guild.members)} members strong!",
                color=self.theme_manager.get_user_theme(str(member.id)).primary
            )
            embed.set_thumbnail(url=member.display_avatar.url)
            await welcome_channel.send(embed=embed)

if __name__ == '__main__':
    bot = ClaudeCordBot()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
