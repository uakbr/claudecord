from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: Path, backup_dir: Path):
        self.config_path = config_path
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.defaults = {
            "bot": {
                "command_prefix": ">",
                "description": "A Discord bot powered by Claude 3.5 Sonnet",
                "status_message": "Ready to chat!",
                "color_theme": 0xda7756,
            },
            "claude": {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 4096,
                "temperature": 0.7,
                "max_memory": 20,
                "system_prompt": """You are a helpful AI assistant. You provide clear, accurate, 
                and engaging responses while maintaining a friendly tone."""
            },
            "database": {
                "path": "data/conversations.db",
                "backup_interval": 86400,  # 24 hours
                "max_attachment_size": 100_000_000,  # 100MB
                "cleanup_interval": 604800  # 7 days
            },
            "limits": {
                "max_conversation_length": 50,
                "rate_limit": 5,  # messages per second
                "max_tokens_per_request": 4096,
                "max_image_size": 20_000_000  # 20MB
            }
        }
        
    def load(self) -> Dict[str, Any]:
        """Load configuration with fallback to defaults"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                user_config = yaml.safe_load(f)
                return self._merge_configs(self.defaults, user_config)
        return self.defaults
    
    def save(self, config: Dict[str, Any]) -> None:
        """Save configuration and create backup"""
        try:
            # Create backup first
            if self.config_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_dir / f"config_backup_{timestamp}.yaml"
                shutil.copy2(self.config_path, backup_path)
                
                # Keep only last 5 backups
                backups = sorted(self.backup_dir.glob("config_backup_*.yaml"))
                for old_backup in backups[:-5]:
                    old_backup.unlink()
            
            # Save new config
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Deep merge of default and user configurations"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result 