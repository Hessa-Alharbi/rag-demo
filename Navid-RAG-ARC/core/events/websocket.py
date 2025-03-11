import json
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from loguru import logger
import redis.asyncio as aioredis
from redis import Redis
from core.settings import get_settings

settings = get_settings()
redis_instance = None

async def get_redis():
    """Get or create Redis connection"""
    global redis_instance
    if redis_instance is None:
        try:
            logger.info(f"Connecting to Redis at {settings.REDIS_URL}")
            redis_instance = await aioredis.from_url(
                settings.REDIS_URL, 
                encoding="utf-8", 
                decode_responses=True
            )
            # Test the connection
            await redis_instance.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    return redis_instance

def send_status_update(user_id: Optional[str], data: Dict[str, Any]) -> bool:
    """
    Send status updates to clients via Redis pub/sub
    This function is called from Celery tasks
    
    Returns:
        bool: True if the message was sent successfully
    """
    try:
        if not user_id:
            logger.warning("No user_id provided for status update")
            return False
        
        # Create a channel specific to the user if user_id is provided
        channel = f"user:{user_id}:tasks" if user_id else "tasks:all"
        
        # Use synchronous Redis client for Celery tasks
        r = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        
        # Add timestamp to data
        import datetime
        data["timestamp"] = datetime.datetime.now().isoformat()
        
        message = json.dumps({
            "type": "task_update",
            "data": data
        })
        
        # Publish message
        result = r.publish(channel, message)
        
        # Log success or failure
        if result > 0:
            logger.info(f"Published event to {channel}: {data['status']} (received by {result} subscribers)")
        else:
            logger.warning(f"Published event to {channel}: {data['status']} (no subscribers)")
        
        # Also store recent messages for retrieval if necessary
        try:
            # Store last 50 messages per user
            key = f"user:{user_id}:recent_messages"
            r.lpush(key, message)
            r.ltrim(key, 0, 49)  # Keep only most recent 50
            r.expire(key, 3600)  # Expire after 1 hour
        except:
            pass
            
        return result > 0
            
    except Exception as e:
        logger.error(f"Error sending status update: {str(e)}", exc_info=True)
        return False

async def send_async_status_update(user_id: Optional[str], data: Dict[str, Any]) -> bool:
    """
    Send status updates to clients via Redis pub/sub
    Async version for use in FastAPI endpoints
    
    Returns:
        bool: True if the message was sent successfully
    """
    try:
        if not user_id:
            logger.warning("No user_id provided for async status update")
            return False
            
        # Create a channel specific to the user if user_id is provided
        channel = f"user:{user_id}:tasks" if user_id else "tasks:all"
        
        # Add timestamp to data
        import datetime
        data["timestamp"] = datetime.datetime.now().isoformat()
        
        redis = await get_redis()
        message = json.dumps({
            "type": "task_update",
            "data": data
        })
        
        # Publish message
        result = await redis.publish(channel, message)
        
        # Log success or failure
        if result > 0:
            logger.info(f"Published async event to {channel}: {data['status']} (received by {result} subscribers)")
        else:
            logger.warning(f"Published async event to {channel}: {data['status']} (no subscribers)")
        
        # Also store recent messages
        try:
            # Store last 50 messages per user
            key = f"user:{user_id}:recent_messages"
            await redis.lpush(key, message)
            await redis.ltrim(key, 0, 49)  # Keep only most recent 50
            await redis.expire(key, 3600)  # Expire after 1 hour
        except:
            pass
            
        return result > 0
            
    except Exception as e:
        logger.error(f"Error sending async status update: {str(e)}", exc_info=True)
        return False

async def subscribe_to_updates(user_id: Optional[str]) -> Tuple[Any, str]:
    """
    Subscribe to status updates
    Returns the pubsub object and the channel name
    """
    try:
        if not user_id:
            logger.warning("No user_id provided for subscription")
            channel = "tasks:all"
        else:
            # Create a channel specific to the user
            channel = f"user:{user_id}:tasks"
            
        redis = await get_redis()
        
        # Create a new PubSub connection for this subscription
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        logger.info(f"Subscribed to Redis channel: {channel}")
        
        # Send any recent messages that might have been missed
        if user_id:
            try:
                key = f"user:{user_id}:recent_messages"
                recent_messages = await redis.lrange(key, 0, 10)  # Get 10 most recent
                
                if recent_messages:
                    logger.info(f"Found {len(recent_messages)} recent messages for user {user_id}")
                    
                    # Add these to the pubsub queue
                    for msg in reversed(recent_messages):  # Oldest first
                        await redis.publish(channel, msg)
            except:
                pass
        
        return pubsub, channel
    except Exception as e:
        logger.error(f"Error subscribing to updates: {str(e)}", exc_info=True)
        raise

async def test_redis_connection() -> bool:
    """Test Redis connection and pubsub functionality"""
    try:
        # Test basic Redis connection
        redis = await get_redis()
        await redis.set("test_key", "test_value")
        value = await redis.get("test_key")
        logger.info(f"Redis connection test: {value}")
        
        # Test pubsub
        channel = "test_channel"
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        
        # Publish a test message
        await redis.publish(channel, json.dumps({"test": "message"}))
        
        # Try to receive the message
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if message:
            logger.info(f"Redis pubsub test successful: {message}")
        else:
            logger.warning("Redis pubsub test: No message received")
            
        # Clean up
        await pubsub.unsubscribe(channel)
        await redis.delete("test_key")
        
        return True
    except Exception as e:
        logger.error(f"Redis connection test failed: {str(e)}", exc_info=True)
        return False

async def get_recent_messages(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent messages for a user"""
    try:
        redis = await get_redis()
        key = f"user:{user_id}:recent_messages"
        messages = await redis.lrange(key, 0, limit-1)
        
        result = []
        for msg in messages:
            try:
                parsed = json.loads(msg)
                result.append(parsed)
            except:
                continue
                
        return result
    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        return []
