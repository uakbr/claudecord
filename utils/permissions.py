from enum import IntEnum
from typing import Optional
from discord import Member, Role
import logging
from discord.ext import commands

logger = logging.getLogger(__name__)

class PermissionLevel(IntEnum):
    USER = 0
    MODERATOR = 1
    ADMIN = 2
    OWNER = 3

class PermissionManager:
    def __init__(self, admin_role_name: str, mod_role_name: str):
        self.admin_role = admin_role_name
        self.mod_role = mod_role_name

    def get_permission_level(self, member: Member) -> PermissionLevel:
        """Get the permission level of a member"""
        if member.guild.owner_id == member.id:
            return PermissionLevel.OWNER
            
        roles = {role.name for role in member.roles}
        
        if self.admin_role in roles:
            return PermissionLevel.ADMIN
        elif self.mod_role in roles:
            return PermissionLevel.MODERATOR
        return PermissionLevel.USER

    def requires_permission(self, level: PermissionLevel):
        """Decorator to check permission level for commands"""
        async def predicate(ctx):
            user_level = self.get_permission_level(ctx.author)
            if user_level >= level:
                return True
            await ctx.send(f"You need {level.name} permissions to use this command.")
            return False
        return commands.check(predicate) 