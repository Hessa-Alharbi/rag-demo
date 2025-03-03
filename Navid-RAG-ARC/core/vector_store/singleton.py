from typing import Optional, Dict
from core.vector_store.manager import VectorStoreManager

class VectorStoreSingleton:
    _instance: Optional[VectorStoreManager] = None
    _conversation_stores: Dict[str, VectorStoreManager] = {}

    @classmethod
    async def get_instance(cls) -> VectorStoreManager:
        if not cls._instance:
            cls._instance = VectorStoreManager()
            await cls._instance.initialize()
        return cls._instance

    @classmethod
    async def get_conversation_store(cls, conversation_id: str) -> VectorStoreManager:
        if conversation_id not in cls._conversation_stores:
            store = VectorStoreManager()
            await store.initialize(namespace=conversation_id)
            cls._conversation_stores[conversation_id] = store
        return cls._conversation_stores[conversation_id]
