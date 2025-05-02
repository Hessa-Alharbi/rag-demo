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
from loguru import logger
import traceback

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

        # Add debug logging
        stack_trace = traceback.format_stack()
        caller_info = stack_trace[-2] if len(stack_trace) > 1 else "Unknown caller"
        logger.debug(f"Creating LLM with provider: {provider}, model: {settings.LLM_MODEL}")
        logger.debug(f"LLM creation called from: {caller_info}")

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model_name=settings.LLM_MODEL, 
                temperature=0.3,  # Lower temperature for more focused outputs
                max_tokens=1500,   # Control length of response
                **config
            )
        elif provider == "cohere":
            from langchain_community.llms import VLLMOpenAI

            return VLLMOpenAI(
                openai_api_key=settings.LLM_API_KEY,
                openai_api_base=settings.LLM_BASE_URL,
                model_name=settings.LLM_MODEL,
                temperature=0.3,  # Lower temperature
                max_tokens=1500,  # Control output length
                **config,
            )
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint
            import os
            
            # Get Arabic-specific configuration from environment
            use_json_response = os.environ.get("JSON_RESPONSE_FORMAT", "true").lower() == "true"
            max_tokens = int(os.environ.get("GEMMA_MAX_TOKENS", "800"))
            
            # Generation parameters specifically tuned for Gemma to work better with Arabic
            generation_params = {
                "temperature": 0.3,       # Higher temperature for more diverse Arabic responses
                "max_new_tokens": max_tokens,
                "repetition_penalty": 1.3, # Increased repetition penalty to prevent repetition in Arabic
                "do_sample": True,        # Enable sampling
                "top_p": 0.5,           # Consider more diverse tokens (increased from 0.9)
                "top_k": 80,             # Consider more tokens at each step for Arabic vocabulary (increased from 50)
                "return_full_text": False, # Only return new text
                "no_repeat_ngram_size": 3, # Prevent 3-grams from repeating
                "early_stopping": True,
                "frequency_penalty": 0.4,  # Increased from 0.3
                "use_cache": True,
                # Important additions for Arabic language handling:
                "include_prompt_in_output": False,
                "strip_whitespace": True
            }
            
            # If we need JSON output, adjust parameters
            if use_json_response:
                generation_params["temperature"] = 0.2  # Very low temperature for structured output
                generation_params["frequency_penalty"] = 0.5  # Higher penalty for JSON to be well-formed
            
            # Update with any user provided config
            if config:
                generation_params.update(config)
            
            logger.info(f"Creating HuggingFace endpoint with model: {settings.LLM_MODEL}")
            
            # Check if we're using Gemma models, which need special handling for Arabic
            if "gemma" in settings.LLM_MODEL.lower():
                logger.info("Using Gemma model with Arabic optimizations")
                
                # Update generation parameters for Arabic
                generation_params.update({
                    "temperature": 0.85,  # Higher temperature for more fluent Arabic
                    "max_new_tokens": int(os.environ.get("GEMMA_MAX_TOKENS", "1200")),  # More tokens for Arabic
                    "top_p": 0.95,  # Increased diversity
                    "top_k": 100,   # Consider more tokens
                    "repetition_penalty": 1.5  # Stronger penalties to avoid repetition issues
                })
                
                # For Gemma models, we need to ensure better Arabic handling
                return HuggingFaceEndpoint(
                    endpoint_url=settings.LLM_BASE_URL,  # استخدام endpoint_url بدلاً من repo_id
                    huggingfacehub_api_token=settings.HF_TOKEN,
                    task="text-generation",
                    client_settings={"timeout": int(os.environ.get("GEMMA_TIMEOUT", "60"))},  # Increased timeout further
                    model_kwargs={
                        "stop": ["</answer>", "</response>", "Human:", "User:", "Question:"],
                        "pad_token_id": 0,
                    },
                    **generation_params
                )
            else:
                # Default for other HuggingFace models
                return HuggingFaceEndpoint(
                    endpoint_url=settings.LLM_BASE_URL,  # استخدام endpoint_url بدلاً من repo_id
                    huggingfacehub_api_token=settings.HF_TOKEN,
                    task="text-generation",
                    client_settings={"timeout": int(os.environ.get("GEMMA_TIMEOUT", "30"))},
                    **generation_params
                )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# client = OpenAI(
#     base_url="https://exclusive-mere-advocate-gotten.trycloudflare.com/v1",
#     api_key="a3f9b8cfd34a89e5e2dc4a9b1d8fde3",  # Your API key
# )

# CohereForAI/aya-expanse-32b
