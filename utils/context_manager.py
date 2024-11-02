from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
import json
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConversationContext:
    """Manages individual conversation contexts with memory management"""
    
    def __init__(self, max_context_length: int = 10, max_age_hours: int = 24):
        """
        Initialize conversation context
        
        Args:
            max_context_length: Maximum number of messages to keep
            max_age_hours: Maximum age of messages before cleanup
        """
        self.max_length = max_context_length
        self.max_age = timedelta(hours=max_age_hours)
        self.context: List[Dict] = []
        self.metadata: Dict = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_messages": 0,
            "summary": None,
            "tokens_used": 0,
            "last_cleanup": datetime.now().isoformat()
        }
        self._lock = asyncio.Lock()
        
    async def add_message(self, role: str, content: str, 
                         token_count: int = 0,
                         metadata: Optional[Dict] = None) -> None:
        """
        Add a message to the context with thread safety
        
        Args:
            role: Message role (user/assistant)
            content: Message content
            token_count: Number of tokens in message
            metadata: Additional message metadata
        """
        async with self._lock:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "tokens": token_count,
                "metadata": metadata or {}
            }
            
            self.context.append(message)
            self.metadata["tokens_used"] += token_count
            
            if len(self.context) > self.max_length:
                removed = self.context.pop(0)
                self.metadata["tokens_used"] -= removed.get("tokens", 0)
                
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["total_messages"] += 1
            
    async def get_context(self, num_messages: Optional[int] = None,
                         max_tokens: Optional[int] = None) -> List[Dict]:
        """
        Get recent context messages with token limit
        
        Args:
            num_messages: Number of recent messages to return
            max_tokens: Maximum total tokens to return
            
        Returns:
            List of context messages
        """
        async with self._lock:
            if max_tokens is not None:
                messages = []
                token_count = 0
                
                for msg in reversed(self.context):
                    msg_tokens = msg.get("tokens", 0)
                    if token_count + msg_tokens > max_tokens:
                        break
                    messages.append(msg)
                    token_count += msg_tokens
                    
                return list(reversed(messages))
            
            if num_messages is not None:
                return self.context[-num_messages:]
            return self.context
    
    async def cleanup_old_messages(self) -> int:
        """
        Remove messages older than max_age
        
        Returns:
            Number of messages removed
        """
        async with self._lock:
            now = datetime.now()
            cutoff = now - self.max_age
            
            original_length = len(self.context)
            self.context = [
                msg for msg in self.context
                if datetime.fromisoformat(msg["timestamp"]) > cutoff
            ]
            
            removed_count = original_length - len(self.context)
            if removed_count > 0:
                self.metadata["last_cleanup"] = now.isoformat()
                logger.debug(f"Removed {removed_count} old messages from context")
                
            return removed_count
    
    async def clear_context(self) -> None:
        """Clear the conversation context"""
        async with self._lock:
            self.context = []
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["tokens_used"] = 0
            
    def set_summary(self, summary: str) -> None:
        """Set conversation summary"""
        self.metadata["summary"] = summary
        
    def to_dict(self) -> Dict:
        """Convert context to dictionary"""
        return {
            "context": self.context,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationContext':
        """Create context from dictionary"""
        context = cls()
        context.context = data["context"]
        context.metadata = data["metadata"]
        return context

class ContextManager:
    """Manages all conversation contexts with periodic cleanup"""
    
    def __init__(self, storage, cleanup_interval: int = 3600):
        """
        Initialize context manager
        
        Args:
            storage: Database storage instance
            cleanup_interval: Seconds between cleanup runs
        """
        self.storage = storage
        self.cleanup_interval = cleanup_interval
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.context_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start periodic cleanup"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop periodic cleanup and save all contexts"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Save all active contexts
        for user_id in list(self.active_contexts.keys()):
            await self.save_context(user_id)
            
    async def get_context(self, user_id: str) -> ConversationContext:
        """
        Get or create context for user with thread safety
        
        Args:
            user_id: Discord user ID
            
        Returns:
            User's conversation context
        """
        try:
            async with self.context_locks[user_id]:
                if user_id not in self.active_contexts:
                    context_data = await self.storage.get_user_context(user_id)
                    if context_data:
                        self.active_contexts[user_id] = ConversationContext.from_dict(context_data)
                    else:
                        self.active_contexts[user_id] = ConversationContext()
                return self.active_contexts[user_id]
        except Exception as e:
            logger.error(f"Error getting context for user {user_id}: {e}")
            return ConversationContext()  # Return empty context on error
    
    async def save_context(self, user_id: str) -> None:
        """Save context to storage with error handling"""
        async with self.context_locks[user_id]:
            if user_id in self.active_contexts:
                try:
                    context_data = self.active_contexts[user_id].to_dict()
                    await self.storage.save_user_context(user_id, context_data)
                except Exception as e:
                    logger.error(f"Error saving context for user {user_id}: {e}")
                    
    async def _cleanup_loop(self):
        """Periodic cleanup of old messages and inactive contexts"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Cleanup old messages in each context
                for user_id, context in list(self.active_contexts.items()):
                    try:
                        async with self.context_locks[user_id]:
                            removed = await context.cleanup_old_messages()
                            if removed > 0:
                                await self.save_context(user_id)
                    except Exception as e:
                        logger.error(f"Error cleaning up context for user {user_id}: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in context cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying