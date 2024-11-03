from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class Theme(Enum):
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    NORD = "nord"
    SOLARIZED = "solarized"
    DRACULA = "dracula"
    MONOKAI = "monokai"

@dataclass
class ThemeColors:
    primary: int
    secondary: int
    success: int
    error: int
    warning: int
    info: int
    background: int
    text: int

THEMES = {
    Theme.DEFAULT: ThemeColors(
        primary=0xda7756,
        secondary=0x7289da,
        success=0x43b581,
        error=0xf04747,
        warning=0xfaa61a,
        info=0x00b0f4,
        background=0x36393f,
        text=0xffffff
    ),
    Theme.DARK: ThemeColors(
        primary=0x7289da,
        secondary=0x99aab5,
        success=0x43b581,
        error=0xf04747,
        warning=0xfaa61a,
        info=0x00b0f4,
        background=0x2c2f33,
        text=0xffffff
    ),
    Theme.LIGHT: ThemeColors(
        primary=0x5865f2,
        secondary=0xeb459e,
        success=0x57f287,
        error=0xed4245,
        warning=0xfee75c,
        info=0x5865f2,
        background=0xffffff,
        text=0x2c2f33
    ),
    Theme.NORD: ThemeColors(
        primary=0x88c0d0,
        secondary=0x81a1c1,
        success=0xa3be8c,
        error=0xbf616a,
        warning=0xebcb8b,
        info=0x5e81ac,
        background=0x2e3440,
        text=0xeceff4
    ),
    Theme.SOLARIZED: ThemeColors(
        primary=0x268bd2,
        secondary=0x2aa198,
        success=0x859900,
        error=0xdc322f,
        warning=0xb58900,
        info=0x268bd2,
        background=0x002b36,
        text=0x839496
    ),
    Theme.DRACULA: ThemeColors(
        primary=0xbd93f9,
        secondary=0xff79c6,
        success=0x50fa7b,
        error=0xff5555,
        warning=0xf1fa8c,
        info=0x8be9fd,
        background=0x282a36,
        text=0xf8f8f2
    ),
    Theme.MONOKAI: ThemeColors(
        primary=0xa6e22e,
        secondary=0xf92672,
        success=0xa6e22e,
        error=0xf92672,
        warning=0xe6db74,
        info=0x66d9ef,
        background=0x272822,
        text=0xf8f8f2
    )
}

class ThemeManager:
    """Manages user theme preferences with persistence"""
    
    def __init__(self):
        self.user_themes = {}
        self.valid_colors = range(0x000000, 0xFFFFFF + 1)
        
    def set_user_theme(self, user_id: str, theme: Theme) -> None:
        """Set user's theme preference"""
        self.user_themes[user_id] = theme
        
    def get_user_theme(self, user_id: str) -> ThemeColors:
        """
        Get user's theme colors
        
        Args:
            user_id: Discord user ID
            
        Returns:
            ThemeColors instance for the user's theme
        """
        theme = self.user_themes.get(user_id, Theme.DEFAULT)
        return THEMES[theme]
        
    def validate_theme(self, theme_name: str) -> Optional[Theme]:
        """
        Validate theme name
        
        Args:
            theme_name: Name of theme to validate
            
        Returns:
            Theme enum value if valid, None otherwise
        """
        try:
            return Theme[theme_name.upper()]
        except KeyError:
            return None
            
    def validate_theme_colors(self, colors: ThemeColors) -> bool:
        return all(
            color in self.valid_colors
            for color in [colors.primary, colors.secondary, colors.text]
        )
        
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.user_themes.clear()