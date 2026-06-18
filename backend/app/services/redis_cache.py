import json
import logging
from typing import Optional, Any
import redis.asyncio as aioredis
from backend.app.config import settings

logger = logging.getLogger("app.redis_cache")

class RedisCache:
    def __init__(self):
        self.host = settings.redis.host
        self.port = settings.redis.port
        self.db = settings.redis.db
        self.redis_client = None
        self.connect()

    def connect(self):
        try:
            url = f"redis://{self.host}:{self.port}/{self.db}"
            self.redis_client = aioredis.from_url(url, encoding="utf-8", decode_responses=True)
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}. Session caches will run in-memory.")

    async def get(self, key: str) -> Optional[Any]:
        """Gets value from cache."""
        if not self.redis_client:
            return None
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
        return None

    async def set(self, key: str, value: Any, expire_seconds: int = 3600) -> bool:
        """Sets value in cache with expiration."""
        if not self.redis_client:
            return False
        try:
            serialized = json.dumps(value)
            await self.redis_client.set(key, serialized, ex=expire_seconds)
            return True
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Deletes key from cache."""
        if not self.redis_client:
            return False
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed for key {key}: {e}")
            return False

redis_cache = RedisCache()
