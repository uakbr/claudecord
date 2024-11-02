from discord import ButtonStyle, SelectOption, Interaction, Embed
from discord.ui import Button, Select, View
from typing import List, Callable, Any, Optional, Dict
import asyncio
import logging
from .themes import Theme, ThemeColors
from .embed_manager import FormatStyle

logger = logging.getLogger(__name__)

class InteractiveView(View):
    """Base class for interactive Discord UI components"""
    
    def __init__(self, timeout: float = 180):
        """
        Initialize the interactive view
        
        Args:
            timeout: Time in seconds before the view becomes inactive
        """
        super().__init__(timeout=timeout)
        self.value = None
        self.response = None
        self._error_handler: Optional[Callable] = None
        
    async def on_timeout(self) -> None:
        """Handle view timeout by disabling all components"""
        try:
            for item in self.children:
                item.disabled = True
            if self.message:
                await self.message.edit(view=self)
        except Exception as e:
            logger.error(f"Error handling timeout: {e}")
            
    def set_error_handler(self, handler: Callable) -> None:
        """Set custom error handler for the view"""
        self._error_handler = handler
        
    async def _handle_error(self, error: Exception, interaction: Interaction) -> None:
        """Internal error handling"""
        if self._error_handler:
            await self._error_handler(error, interaction)
        else:
            logger.error(f"Interaction error: {error}")
            try:
                await interaction.response.send_message(
                    "An error occurred while processing your interaction.",
                    ephemeral=True
                )
            except Exception as e:
                logger.error(f"Error sending error message: {e}")

class ConfirmationView(InteractiveView):
    """View for confirmation dialogs"""
    
    def __init__(self, *, 
                 confirm_label: str = "Confirm",
                 cancel_label: str = "Cancel",
                 timeout: float = 180):
        """
        Initialize confirmation view
        
        Args:
            confirm_label: Label for confirm button
            cancel_label: Label for cancel button
            timeout: View timeout in seconds
        """
        super().__init__(timeout=timeout)
        
        self.add_item(Button(
            label=confirm_label,
            style=ButtonStyle.green,
            custom_id="confirm"
        ))
        self.add_item(Button(
            label=cancel_label,
            style=ButtonStyle.red,
            custom_id="cancel"
        ))

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Handle button interactions"""
        try:
            self.value = interaction.custom_id == "confirm"
            self.stop()
            
            # Disable buttons after selection
            for item in self.children:
                item.disabled = True
            await interaction.response.edit_message(view=self)
            
            return True
        except Exception as e:
            await self._handle_error(e, interaction)
            return False

class PaginationView(InteractiveView):
    """View for paginated content"""
    
    def __init__(self, pages: List[Any], *, timeout: float = 180):
        """
        Initialize pagination view
        
        Args:
            pages: List of page content (usually Embeds)
            timeout: View timeout in seconds
        """
        super().__init__(timeout=timeout)
        self.pages = pages
        self.current_page = 0
        self.max_pages = len(pages)
        
        # Navigation buttons
        self.first_page = Button(
            label="<<", 
            style=ButtonStyle.grey,
            custom_id="first",
            disabled=True
        )
        self.prev_page = Button(
            label="<",
            style=ButtonStyle.grey,
            custom_id="prev",
            disabled=True
        )
        self.page_counter = Button(
            label=f"Page 1/{self.max_pages}",
            style=ButtonStyle.grey,
            custom_id="counter",
            disabled=True
        )
        self.next_page = Button(
            label=">",
            style=ButtonStyle.grey,
            custom_id="next"
        )
        self.last_page = Button(
            label=">>",
            style=ButtonStyle.grey,
            custom_id="last"
        )
        
        self.add_item(self.first_page)
        self.add_item(self.prev_page)
        self.add_item(self.page_counter)
        self.add_item(self.next_page)
        self.add_item(self.last_page)

    def _update_buttons(self) -> None:
        """Update button states based on current page"""
        self.first_page.disabled = self.current_page == 0
        self.prev_page.disabled = self.current_page == 0
        self.next_page.disabled = self.current_page == self.max_pages - 1
        self.last_page.disabled = self.current_page == self.max_pages - 1
        self.page_counter.label = f"Page {self.current_page + 1}/{self.max_pages}"

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Handle navigation button interactions"""
        try:
            if interaction.custom_id == "first":
                self.current_page = 0
            elif interaction.custom_id == "prev":
                self.current_page = max(0, self.current_page - 1)
            elif interaction.custom_id == "next":
                self.current_page = min(len(self.pages) - 1, self.current_page + 1)
            elif interaction.custom_id == "last":
                self.current_page = len(self.pages) - 1
                
            self._update_buttons()
            
            await interaction.response.edit_message(
                embed=self.pages[self.current_page],
                view=self
            )
            return True
        except Exception as e:
            await self._handle_error(e, interaction)
            return False

class SettingsView(InteractiveView):
    """View for user settings configuration"""
    
    def __init__(self, settings: Dict[str, Any], theme_colors: ThemeColors, *, timeout: float = 180):
        """
        Initialize settings view
        
        Args:
            settings: Current user settings
            theme_colors: User's theme colors
            timeout: View timeout in seconds
        """
        super().__init__(timeout=timeout)
        self.settings = settings
        self.theme_colors = theme_colors
        
        # Theme selector
        self.add_item(Select(
            custom_id="theme",
            placeholder="Select a theme",
            options=[
                SelectOption(
                    label=theme.value,
                    value=theme.value,
                    default=settings.get('theme') == theme.value
                )
                for theme in Theme
            ]
        ))
        
        # Format style selector
        self.add_item(Select(
            custom_id="format_style",
            placeholder="Select format style",
            options=[
                SelectOption(
                    label=style.value,
                    value=style.value,
                    default=settings.get('format_style') == style.value
                )
                for style in FormatStyle
            ]
        ))
        
        # Language selector
        self.add_item(Select(
            custom_id="language",
            placeholder="Select preferred language",
            options=[
                SelectOption(label="English", value="en", default=settings.get('language') == 'en'),
                SelectOption(label="Spanish", value="es", default=settings.get('language') == 'es'),
                SelectOption(label="French", value="fr", default=settings.get('language') == 'fr'),
                SelectOption(label="German", value="de", default=settings.get('language') == 'de'),
                SelectOption(label="Japanese", value="ja", default=settings.get('language') == 'ja')
            ]
        ))

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Handle settings selection"""
        try:
            setting_id = interaction.custom_id
            new_value = interaction.values[0]
            
            # Update the setting
            self.value = {setting_id: new_value}
            
            # Create preview embed
            embed = Embed(
                title="Setting Updated",
                description=f"Changed {setting_id} to: {new_value}",
                color=self.theme_colors.primary
            )
            
            await interaction.response.edit_message(embed=embed, view=self)
            return True
        except Exception as e:
            await self._handle_error(e, interaction)
            return False