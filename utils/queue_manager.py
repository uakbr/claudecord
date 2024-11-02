import asyncio
from typing import Dict, Any, Optional, NamedTuple
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    """Rate limit configuration"""
    messages: int
    interval: float  # in seconds
    burst: int = 0   # additional messages allowed in burst

@dataclass
class RateLimitState:
    """Tracks rate limit state"""
    tokens: float = field(default=0.0)
    last_update: float = field(default_factory=time.monotonic)
    waiting: int = field(default=0)

class RateLimiter:
    """Enhanced rate limiter with burst support"""
    
    def __init__(self, rate_limit: RateLimit):
        """
        Initialize rate limiter
        
        Args:
            rate_limit: Rate limit configuration
        """
        self.rate_limit = rate_limit
        self.state = RateLimitState(tokens=float(rate_limit.messages))
        self._lock = asyncio.Lock()
        
    async def acquire(self, amount: float = 1.0) -> bool:
        """
        Acquire rate limit tokens
        
        Args:
            amount: Number of tokens to acquire
            
        Returns:
            bool: True if tokens were acquired, False if rate limited
        """
        async with self._lock:
            now = time.monotonic()
            time_passed = now - self.state.last_update
            
            # Replenish tokens
            self.state.tokens = min(
                float(self.rate_limit.messages + self.rate_limit.burst),
                self.state.tokens + time_passed * (self.rate_limit.messages / self.rate_limit.interval)
            )
            self.state.last_update = now
            
            if self.state.tokens >= amount:
                self.state.tokens -= amount
                return True
                
            # Rate limited
            self.state.waiting += 1
            try:
                wait_time = (amount - self.state.tokens) * (self.rate_limit.interval / self.rate_limit.messages)
                await asyncio.sleep(wait_time)
                self.state.tokens -= amount
                return True
            finally:
                self.state.waiting -= 1

class MessageQueue:
    """Enhanced message queue with multi-level rate limiting"""
    
    def __init__(self, rate_limit: int = 5):
        self.rate_limit = rate_limit
        self.user_queues = {}
        self.server_queues = {}
        self.user_limiters = defaultdict(lambda: RateLimit(5, 1.0))
        self.server_limiters = defaultdict(lambda: RateLimit(10, 1.0))
        self.global_limiter = RateLimit(20, 1.0)
        self.processing = False
        self._cleanup_task = None
        
    async def start(self):
        """Start queue processing and cleanup"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop queue processing"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
    async def add_message(self, user_id: str, server_id: str, message: Dict[str, Any]):
        """
        Add message to queue with multi-level rate limiting
        
        Args:
            user_id: Discord user ID
            server_id: Discord server ID
            message: Message data
        """
        # Create queues if they don't exist
        if user_id not in self.user_queues:
            self.user_queues[user_id] = asyncio.Queue()
        if server_id not in self.server_queues:
            self.server_queues[server_id] = asyncio.Queue()
            
        # Add to both user and server queues
        message_data = {
            'user_id': user_id,
            'server_id': server_id,
            'content': message,
            'timestamp': time.monotonic()
        }
        
        await self.user_queues[user_id].put(message_data)
        await self.server_queues[server_id].put(message_data)
        
        if not self.processing:
            asyncio.create_task(self._process_queues())

    async def _process_queues(self):
        """Process messages with rate limiting"""
        self.processing = True
        try:
            while True:
                processed = False
                
                # Process one message from each queue if possible
                for user_id, user_queue in self.user_queues.items():
                    if not user_queue.empty():
                        message = await user_queue.get()
                        server_id = message['server_id']
                        
                        # Check all rate limits
                        if await self.global_limiter.acquire() and \
                           await self.user_limiters[user_id].acquire() and \
                           await self.server_limiters[server_id].acquire():
                            
                            await self._process_message(message)
                            user_queue.task_done()
                            processed = True
                            
                            # Remove from server queue
                            server_queue = self.server_queues[server_id]
                            while not server_queue.empty():
                                if server_queue.get_nowait() == message:
                                    server_queue.task_done()
                                    break
                
                if not processed and all(q.empty() for q in self.user_queues.values()):
                    break
                    
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error processing message queue: {e}")
        finally:
            self.processing = False

    async def _process_message(self, message: Dict[str, Any]):
        """Process individual message with error handling"""
        try:
            content = message['content']
            await content['callback'](content['msg'], content['content'])
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    async def _cleanup_loop(self):
        """Cleanup expired messages and empty queues"""
        while True:
            try:
                # Remove empty queues
                self.user_queues = {
                    user_id: queue 
                    for user_id, queue in self.user_queues.items() 
                    if not queue.empty()
                }
                self.server_queues = {
                    server_id: queue 
                    for server_id, queue in self.server_queues.items() 
                    if not queue.empty()
                }
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying