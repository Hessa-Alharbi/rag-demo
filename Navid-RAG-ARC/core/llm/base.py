from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        pass

class BaseEmbeddings(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        pass

class BaseVectorStore(ABC):
    @abstractmethod
    async def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        pass
    
    @abstractmethod
    async def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Dict[str, Any]]:
        pass
