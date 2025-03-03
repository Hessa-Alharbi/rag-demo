from typing import Dict, Any, Optional
from core.config import get_settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_chroma import Chroma
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from core.vector_store.connection import MilvusConnectionManager

from langchain_community.vectorstores import FAISS


class ModelFactory:
    @staticmethod
    def create_embeddings(provider: str = None, config: Dict[str, Any] = None):
        settings = get_settings()
        provider = provider or settings.EMBEDDING_PROVIDER
        config = config or settings.EMBEDDING_CONFIG

        if provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL, **config)
        elif provider == "openai":
            return OpenAIEmbeddings(**config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    @staticmethod
    def create_vector_store(
        embeddings,
        provider: Optional[str] = None,
        create_collection: bool = False,
        **kwargs,
    ):
        settings = get_settings()
        provider = provider or settings.VECTOR_STORE_PROVIDER

        if provider == "milvus":
            collection_name = kwargs.pop("collection_name", settings.MILVUS_COLLECTION)

            # Ensure Milvus connection
            MilvusConnectionManager.ensure_connection()

            # Only create collection if explicitly requested
            if create_collection:
                dimension = kwargs.pop("dimension", settings.MILVUS_DIMENSION)
                from core.vector_store.utils import create_milvus_collection

                create_milvus_collection(collection_name, dimension)

            return Milvus(
                embedding_function=embeddings,
                collection_name=collection_name,
                connection_args=settings.milvus_connection_args,
                text_field="text",
                vector_field="embeddings",
                **kwargs,
            )
        elif provider == "chroma":
            return Chroma(embedding_function=embeddings, **kwargs)
        elif provider == "faiss":
            return FAISS(embedding_function=embeddings, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

    @staticmethod
    def create_llm(provider: str = None, config: Dict[str, Any] = None):
        settings = get_settings()
        provider = provider or settings.LLM_PROVIDER
        config = config or settings.LLM_CONFIG

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model_name=settings.LLM_MODEL, temperature=0.7, **config)
        elif provider == "cohere":
            from langchain_community.llms import VLLMOpenAI

            return VLLMOpenAI(
                openai_api_key=settings.LLM_API_KEY,
                openai_api_base=settings.LLM_BASE_URL,
                model_name=settings.LLM_MODEL,
                **config,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# client = OpenAI(
#     base_url="https://exclusive-mere-advocate-gotten.trycloudflare.com/v1",
#     api_key="a3f9b8cfd34a89e5e2dc4a9b1d8fde3",  # Your API key
# )

# CohereForAI/aya-expanse-32b
