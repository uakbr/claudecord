from discord import Embed, Color
from typing import Optional, List, Dict
from enum import Enum
from discord.ext import commands
from .themes import ThemeManager, Theme, ThemeColors

class FormatStyle(Enum):
    """Enum defining different formatting styles for messages"""
    DEFAULT = "default"
    MINIMAL = "minimal"
    DETAILED = "detailed"
    COMPACT = "compact"
    EXPANDED = "expanded"

class EmbedManager:
    """Manages Discord embed creation and formatting with theme support"""
    
    def __init__(self, theme_manager: ThemeManager):
        """
        Initialize the EmbedManager
        
        Args:
            theme_manager: ThemeManager instance for handling user themes
        """
        self.theme_manager = theme_manager
        self.format_preferences: Dict[str, Dict] = {}
        
        # Default formatting templates
        self.templates = {
            FormatStyle.DEFAULT: {
                "code_block": "```{lang}\n{code}\n```",
                "quote": "> {text}",
                "bullet": "• {text}",
                "number": "{num}. {text}",
                "table_header": "| {headers} |",
                "table_row": "| {cells} |",
                "table_separator": "|---" * "{col_count}" + "|"
            },
            FormatStyle.MINIMAL: {
                "code_block": "`{code}`",
                "quote": "{text}",
                "bullet": "- {text}",
                "number": "{num}. {text}",
                "table_header": "{headers}",
                "table_row": "{cells}",
                "table_separator": "---"
            },
            FormatStyle.DETAILED: {
                "code_block": "```{lang}\n# {title}\n{code}\n```",
                "quote": "📝 {text}",
                "bullet": "◆ {text}",
                "number": "#{num} {text}",
                "table_header": "┌─{headers}─┐",
                "table_row": "│ {cells} │",
                "table_separator": "├─" * "{col_count}" + "┤"
            }
        }

    def set_user_preference(self, user_id: str, preferences: Dict) -> None:
        """Set formatting preferences for a user"""
        self.format_preferences[user_id] = preferences

    def get_user_preference(self, user_id: str) -> Dict:
        """Get user's formatting preferences"""
        return self.format_preferences.get(user_id, {})

    def create_response_embed(self, content: str, author_name: str, 
                            author_avatar: Optional[str] = None,
                            user_id: Optional[str] = None,
                            format_style: FormatStyle = FormatStyle.DEFAULT) -> List[Embed]:
        """
        Create formatted response embeds with user's theme
        
        Args:
            content: Message content to format
            author_name: Name of the message author
            author_avatar: URL of author's avatar (optional)
            user_id: Discord user ID for theme/format preferences
            format_style: Default format style to use
            
        Returns:
            List of formatted Discord Embeds
        """
        # Get user's theme colors
        theme_colors = self.theme_manager.get_user_theme(user_id) if user_id else None
        embed_color = theme_colors.primary if theme_colors else 0xda7756
        
        # Apply user preferences if available
        style = self.format_preferences.get(user_id, {}).get('style', format_style)
        template = self.templates[style]
        
        # Format content based on template
        formatted_content = self._format_content(content, template)
        
        # Split into chunks and create embeds
        chunks = [formatted_content[i:i+4096] for i in range(0, len(formatted_content), 4096)]
        embeds = []
        
        for i, chunk in enumerate(chunks):
            embed = Embed(
                description=chunk,
                color=embed_color
            )
            
            # Add footer with page numbers if multiple embeds
            if len(chunks) > 1:
                embed.set_footer(
                    text=f"Page {i+1}/{len(chunks)}",
                    color=theme_colors.secondary if theme_colors else None
                )
            
            if i == 0:  # Only first embed gets author info
                if author_avatar:
                    embed.set_author(
                        name=author_name,
                        icon_url=author_avatar
                    )
                else:
                    embed.set_author(name=author_name)
                    
            embeds.append(embed)
            
        return embeds

    def _format_content(self, content: str, template: Dict) -> str:
        """Format content using template"""
        # Format code blocks
        content = self._format_code_blocks(content, template)
        
        # Format quotes
        content = self._format_quotes(content, template)
        
        # Format lists
        content = self._format_lists(content, template)
        
        # Format tables
        content = self._format_tables(content, template)
        
        return content

    def _format_code_blocks(self, content: str, template: Dict) -> str:
        """Format code blocks in content"""
        import re
        code_block_pattern = r"```(\w+)?\n(.*?)\n```"
        
        def replace_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2)
            return template["code_block"].format(lang=lang, code=code)
            
        return re.sub(code_block_pattern, replace_code_block, content, flags=re.DOTALL)

    def _format_quotes(self, content: str, template: Dict) -> str:
        """Format quotes in content"""
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('>'):
                formatted_lines.append(template["quote"].format(
                    text=line[1:].strip()
                ))
            else:
                formatted_lines.append(line)
                
        return '\n'.join(formatted_lines)

    def _format_lists(self, content: str, template: Dict) -> str:
        """Format bullet points and numbered lists"""
        lines = content.split('\n')
        formatted_lines = []
        number_counter = 1
        
        for line in lines:
            if line.strip().startswith('•'):
                formatted_lines.append(template["bullet"].format(
                    text=line.strip()[1:].strip()
                ))
            elif line.strip().startswith(('1.', '2.', '3.')):
                formatted_lines.append(template["number"].format(
                    num=number_counter,
                    text=line.strip()[2:].strip()
                ))
                number_counter += 1
            else:
                formatted_lines.append(line)
                number_counter = 1
                
        return '\n'.join(formatted_lines)

    def _format_tables(self, content: str, template: Dict) -> str:
        """Format tables in content"""
        import re
        table_pattern = r"\|.*?\|"
        
        def format_table_row(row: str) -> str:
            cells = [cell.strip() for cell in row.split('|')[1:-1]]
            return template["table_row"].format(cells=' | '.join(cells))
        
        lines = content.split('\n')
        formatted_lines = []
        in_table = False
        
        for i, line in enumerate(lines):
            if re.match(table_pattern, line):
                if not in_table:  # Table start
                    in_table = True
                    formatted_lines.append(
                        template["table_header"].format(
                            headers=' | '.join(cell.strip() for cell in line.split('|')[1:-1])
                        )
                    )
                    formatted_lines.append(
                        template["table_separator"].format(
                            col_count=len(line.split('|'))-2
                        )
                    )
                else:  # Table row
                    formatted_lines.append(format_table_row(line))
            else:
                in_table = False
                formatted_lines.append(line)
                
        return '\n'.join(formatted_lines)

    def validate_format_style(self, style: str) -> bool:
        """
        Validate if a format style exists
        
        Args:
            style: Format style string to validate
            
        Returns:
            bool: True if style is valid, False otherwise
        """
        try:
            FormatStyle[style.upper()]
            return True
        except KeyError:
            return False