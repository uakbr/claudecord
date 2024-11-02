from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import logging
import asyncio
from collections import deque
import aiosqlite
import os
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

@dataclass
class Metrics:
    """Stores bot performance metrics"""
    start_time: float = field(default_factory=time.time)
    total_messages: int = 0
    total_errors: int = 0
    command_usage: Dict[str, int] = field(default_factory=dict)
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    active_users: Dict[str, datetime] = field(default_factory=dict)
    rate_limit_hits: int = 0
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    disk_usage: List[float] = field(default_factory=list)
    network_stats: Dict[str, int] = field(default_factory=lambda: {"sent": 0, "received": 0})
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time
    
    @property
    def error_rate(self) -> float:
        if self.total_messages == 0:
            return 0.0
        return (self.total_errors / self.total_messages) * 100
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def avg_cpu_usage(self) -> float:
        if not self.cpu_usage:
            return 0.0
        return sum(self.cpu_usage[-10:]) / min(len(self.cpu_usage), 10)

class MonitoringSystem:
    """Handles bot monitoring and health checks"""
    
    def __init__(self, check_interval: float = 60.0, db_path: str = "conversations.db"):
        self.metrics = Metrics()
        self.check_interval = check_interval
        self.db_path = db_path
        self.health_checks: Dict[str, bool] = {
            "database": True,
            "api": True,
            "memory": True,
            "rate_limits": True
        }
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start the monitoring system"""
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring system started")
        
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Monitoring system stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self._perform_health_checks()
                await self._collect_metrics()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
    async def _perform_health_checks(self):
        """Perform system health checks"""
        # Check database connection
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
            self.health_checks["database"] = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            self.health_checks["database"] = False
            
        # Check API status
        try:
            async with AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY')) as client:
                await client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}]
                )
            self.health_checks["api"] = True
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            self.health_checks["api"] = False
            
        # Check memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_percent = process.memory_percent()
            self.metrics.memory_usage.append(memory_percent)
            self.health_checks["memory"] = memory_percent < 90.0
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            self.health_checks["memory"] = False
            
        # Add rate limit check
        try:
            rate_limit_status = self.metrics.rate_limit_hits < 100  # Threshold for last hour
            self.health_checks["rate_limits"] = rate_limit_status
            if not rate_limit_status:
                logger.warning("Rate limit threshold exceeded")
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            self.health_checks["rate_limits"] = False
            
        # Check error rate
        try:
            error_rate = self.metrics.error_rate
            if error_rate > 10.0:  # Alert if error rate exceeds 10%
                logger.warning(f"High error rate detected: {error_rate:.2f}%")
                # Could trigger additional alerting here
        except Exception as e:
            logger.error(f"Error rate check failed: {e}")
            
        # Performance checks
        try:
            # CPU usage alert
            if self.metrics.avg_cpu_usage > 80:
                logger.warning(f"High CPU usage detected: {self.metrics.avg_cpu_usage:.1f}%")
                
            # Memory usage trend
            if len(self.metrics.memory_usage) >= 5:
                recent_trend = self.metrics.memory_usage[-5:]
                if all(x > y for x, y in zip(recent_trend[1:], recent_trend[:-1])):
                    logger.warning("Memory usage consistently increasing")
                    
            # Disk space alert
            if self.metrics.disk_usage and self.metrics.disk_usage[-1] > 90:
                logger.warning(f"Low disk space: {self.metrics.disk_usage[-1]:.1f}% used")
                
        except Exception as e:
            logger.error(f"Error in performance checks: {e}")
            
    async def _collect_metrics(self):
        """Collect system metrics"""
        now = datetime.now()
        
        try:
            # System metrics
            import psutil
            process = psutil.Process()
            
            # CPU usage
            self.metrics.cpu_usage.append(process.cpu_percent())
            if len(self.metrics.cpu_usage) > 1440:  # Keep 24 hours of data
                self.metrics.cpu_usage = self.metrics.cpu_usage[-1440:]
                
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics.disk_usage.append(disk.percent)
            if len(self.metrics.disk_usage) > 144:  # Keep 12 hours of data
                self.metrics.disk_usage = self.metrics.disk_usage[-144:]
                
            # Network stats
            net_io = psutil.net_io_counters()
            self.metrics.network_stats["sent"] = net_io.bytes_sent
            self.metrics.network_stats["received"] = net_io.bytes_received
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
        # Clean up old active users
        self.metrics.active_users = {
            user_id: last_seen 
            for user_id, last_seen in self.metrics.active_users.items()
            if now - last_seen < timedelta(minutes=15)
        }
        
        # Reset rate limit counter hourly
        if hasattr(self, '_last_rate_limit_reset'):
            if now - self._last_rate_limit_reset > timedelta(hours=1):
                self.metrics.rate_limit_hits = 0
                self._last_rate_limit_reset = now
        else:
            self._last_rate_limit_reset = now
            
        # Trim memory usage history
        if len(self.metrics.memory_usage) > 1440:
            self.metrics.memory_usage = self.metrics.memory_usage[-1440:]
            
    def record_message(self, user_id: str):
        """Record a message from a user"""
        try:
            self.metrics.total_messages += 1
            self.metrics.active_users[user_id] = datetime.now()
        except Exception as e:
            logger.error(f"Error recording message: {e}")
        
    def record_error(self):
        """Record an error occurrence"""
        self.metrics.total_errors += 1
        
    def record_command(self, command_name: str):
        """Record command usage"""
        self.metrics.command_usage[command_name] = \
            self.metrics.command_usage.get(command_name, 0) + 1
        
    def record_response_time(self, response_time: float):
        """Record message response time"""
        self.metrics.response_times.append(response_time)
        
    def record_rate_limit(self):
        """Record rate limit occurrence"""
        self.metrics.rate_limit_hits += 1
        
    def get_health_status(self) -> Dict:
        """Get current health status"""
        return {
            "status": "healthy" if all(self.health_checks.values()) else "unhealthy",
            "checks": self.health_checks,
            "metrics": {
                "uptime": self.metrics.uptime,
                "total_messages": self.metrics.total_messages,
                "error_rate": self.metrics.error_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "active_users": len(self.metrics.active_users),
                "rate_limit_hits": self.metrics.rate_limit_hits,
                "memory_usage": self.metrics.memory_usage[-1] if self.metrics.memory_usage else 0,
                "cpu_usage": self.metrics.avg_cpu_usage,
                "disk_usage": self.metrics.disk_usage[-1] if self.metrics.disk_usage else 0,
                "network": {
                    "sent_mb": self.metrics.network_stats["sent"] / 1024 / 1024,
                    "received_mb": self.metrics.network_stats["received"] / 1024 / 1024
                }
            }
        } 