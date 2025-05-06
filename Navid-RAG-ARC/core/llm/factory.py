from typing import Dict, Any, Optional
from core.config import get_settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_milvus import Milvus
from langchain_chroma import Chroma
from openai import OpenAI
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from core.vector_store.connection import MilvusConnectionManager
from loguru import logger
import traceback
import os
import contextlib
import httpx

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
        """
        إنشاء نموذج LLM للاتصال بـ yehia-7b-preview-red.
        يستخدم OpenAI API للاتصال بنموذج yehia عبر واجهة متوافقة.
        """
        settings = get_settings()
        provider = provider or settings.LLM_PROVIDER
        config = config or settings.LLM_CONFIG
        
        logger.info(f"===== CREATING LLM MODEL =====")
        logger.info(f"Provider: {provider}, Model: {settings.LLM_MODEL}")
        logger.info(f"Base URL: {settings.LLM_BASE_URL}")
        
        # التحقق من وجود مفتاح API
        api_key = settings.OPENAI_API_KEY or settings.HF_TOKEN
        if not api_key:
            raise ValueError(f"API key not found. Please set OPENAI_API_KEY or HF_TOKEN in .env file.")
        
        logger.info(f"API key found: {'yes' if api_key else 'no'}")
        
        try:
            # إنشاء عميل HTTP مخصص مع رؤوس التفويض
            http_client = httpx.Client(
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0
            )
            
            # إنشاء عميل OpenAI مخصص
            custom_client = OpenAI(
                api_key="sk-dummy",  # سيتم تجاهله لأننا نستخدم عميل HTTP مخصص
                base_url=settings.LLM_BASE_URL,
                http_client=http_client
            )
            
            # إنشاء نموذج ChatOpenAI الخاص بـ LangChain
            model = ChatOpenAI(
                model="tgi",  # اسم النموذج المستخدم في الخادم
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS, 
                openai_api_base=settings.LLM_BASE_URL,
                openai_api_key=api_key,
                client=custom_client
            )
            
            # إضافة خاصية model_name للتوافق
            model.model_name = "yehia-7b-preview-red"
            
            logger.info(f"Successfully created LLM model: yehia-7b-preview-red")
            return model
            
        except Exception as e:
            logger.error(f"Error creating LLM: {str(e)}")
            logger.error(traceback.format_exc())
            raise


# client = OpenAI(
#     base_url="https://exclusive-mere-advocate-gotten.trycloudflare.com/v1",
#     api_key="a3f9b8cfd34a89e5e2dc4a9b1d8fde3",  # Your API key
# )

# CohereForAI/aya-expanse-32b
