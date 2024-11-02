from typing import List, Dict, Optional, Tuple
import aiosqlite
import json
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationStorage:
    def __init__(self, db_path: str):
        self.db_path: str = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id TEXT PRIMARY KEY,
                    history TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    filename TEXT,
                    content BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_attachments_user ON attachments(user_id)")
            await db.execute("""
                DELETE FROM attachments 
                WHERE created_at < datetime('now', '-7 days')
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_contexts (
                    user_id TEXT PRIMARY KEY,
                    context TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    theme TEXT,
                    format_style TEXT,
                    language TEXT,
                    voice TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_themes (
                    user_id TEXT PRIMARY KEY,
                    theme TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated_at 
                ON user_contexts(updated_at)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_preferences_user 
                ON user_preferences(user_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_preferences_updated 
                ON user_preferences(updated_at)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_themes_updated 
                ON user_themes(updated_at)
            """)
            
            await db.commit()

    async def get_convo(self, user_id: str) -> List[Dict[str, any]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT history FROM conversations WHERE user_id = ?", (user_id,)) as cursor:
                result = await cursor.fetchone()
                if result:
                    return json.loads(result[0])
        return []

    async def update_convo(self, user_id: str, conversation: List[Dict[str, any]]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO conversations (user_id, history) VALUES (?, ?)",
                (user_id, json.dumps(conversation))
            )
            await db.commit()

    async def store_attachment(self, user_id: str, filename: str, content: bytes) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO attachments (user_id, filename, content) VALUES (?, ?, ?)",
                (user_id, filename, content)
            )
            await db.commit()
            return cursor.lastrowid

    async def get_attachment(self, attachment_id: int) -> Tuple[str, bytes]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT filename, content FROM attachments WHERE id = ?", (attachment_id,)) as cursor:
                result = await cursor.fetchone()
                if result:
                    return result
        return None

    async def delete_user_convo(self, user_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM attachments WHERE user_id = ?", (user_id,))
            await db.commit()

    async def update_user_settings(self, user_id: str, settings: Dict[str, any]) -> None:
        """Update user-specific settings"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    settings TEXT NOT NULL
                )
            """)
            
            await db.execute(
                "INSERT OR REPLACE INTO user_settings (user_id, settings) VALUES (?, ?)",
                (user_id, json.dumps(settings))
            )
            await db.commit()

    async def get_user_settings(self, user_id: str) -> Dict[str, any]:
        """Get user-specific settings"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT settings FROM user_settings WHERE user_id = ?", 
                (user_id,)
            ) as cursor:
                result = await cursor.fetchone()
                return json.loads(result[0]) if result else {}

    async def save_user_context(self, user_id: str, context_data: Dict) -> None:
        """Save user context to database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO user_contexts 
                   (user_id, context, updated_at) 
                   VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (user_id, json.dumps(context_data))
            )
            await db.commit()

    async def get_user_context(self, user_id: str) -> Optional[Dict]:
        """Get user context from database"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT context FROM user_contexts WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                result = await cursor.fetchone()
                return json.loads(result[0]) if result else None

    async def save_user_theme(self, user_id: str, theme: str) -> None:
        """Save user theme preference"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO user_themes (user_id, theme, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (user_id, theme))
            await db.commit()
            
    async def get_all_themes(self):
        """Get all user themes"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT user_id, theme FROM user_themes"
            ) as cursor:
                async for row in cursor:
                    yield row[0], row[1]

    async def cleanup_old_data(self):
        """Periodically cleanup old data"""
        async with aiosqlite.connect(self.db_path) as db:
            # Delete old attachments
            await db.execute("""
                DELETE FROM attachments 
                WHERE created_at < datetime('now', '-7 days')
            """)
            
            # Delete old contexts
            await db.execute("""
                DELETE FROM user_contexts 
                WHERE updated_at < datetime('now', '-30 days')
            """)
            
            await db.commit()
