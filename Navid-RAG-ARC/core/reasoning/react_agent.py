"""
ReACT (Reasoning and Action) agent implementation for improved RAG response generation
with step-by-step reasoning and explicit thought processes.
"""
from typing import Dict, List, Any, Optional
import re
import asyncio
from loguru import logger
from langdetect import detect
from core.llm.factory import ModelFactory
from core.llm.prompt_templates import REACT_REASONING_TEMPLATE, CROSS_VALIDATION_TEMPLATE
from core.language.arabic_utils import ArabicTextProcessor


class ReACTAgent:
    """
    Agent implementing the ReACT (Reasoning and Action) pattern for
    multi-step reasoning over retrieved context.
    """
    def __init__(self):
        self.llm = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the agent with required models."""
        if self._initialized:
            return
            
        async with self._init_lock:
            if not self._initialized:
                self.llm = ModelFactory.create_llm()
                self._initialized = True
                logger.info("ReACT agent initialized")
    
    async def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the ReACT pattern with step-by-step reasoning.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            conversation_history: Previous conversation turns
            
        Returns:
            Dict containing the response, reasoning steps, and related metadata
        """
        await self.initialize()
        
        # Detect language for appropriate response format
        try:
            language = detect(query)
        except:
            language = "en"
            
        is_arabic = language == "ar" or ArabicTextProcessor.contains_arabic(query)
        
        # Format context for the prompt
        formatted_context = self._format_context(context_docs)
        
        # Generate response with ReACT pattern
        prompt = REACT_REASONING_TEMPLATE.format(
            context=formatted_context,
            question=query
        )
        
        try:
            response = await self.llm.agenerate([prompt])
            full_response = response.generations[0][0].text.strip()
            
            # Extract reasoning steps and final answer
            thought_sections, final_answer = self._parse_reasoning(full_response)
            
            # Cross-validate the final answer with the context
            validated_answer = await self._validate_answer(
                query=query, 
                response=final_answer, 
                context_docs=context_docs
            )
            
            # For Arabic, ensure natural-sounding text
            if is_arabic and validated_answer:
                validated_answer = await self._refine_arabic_text(query, validated_answer)
                
            result = {
                "response": validated_answer or final_answer,
                "thought_process": thought_sections,
                "full_reasoning": full_response,
                "is_arabic": is_arabic
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating ReACT response: {e}")
            # Provide graceful fallback
            if is_arabic:
                return {
                    "response": "عذراً، لم أتمكن من معالجة هذا السؤال. يرجى إعادة صياغة سؤالك.",
                    "thought_process": [],
                    "is_arabic": True
                }
            else:
                return {
                    "response": "I'm sorry, I couldn't process this question. Please try rephrasing your question.",
                    "thought_process": [],
                    "is_arabic": False
                }
    
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format context documents into a string suitable for the prompt."""
        if not context_docs:
            return "No relevant context documents available."
            
        formatted_chunks = []
        
        for i, doc in enumerate(context_docs):
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Document")
            title = metadata.get("title", f"Document {i+1}")
            
            formatted_chunk = f"Document {i+1}: {title}\nSource: {source}\n{content}\n"
            formatted_chunks.append(formatted_chunk)
            
        return "\n\n".join(formatted_chunks)
    
    def _parse_reasoning(self, full_response: str) -> tuple:
        """
        Parse the reasoning steps and final answer from the ReACT response.
        
        Returns:
            tuple: (thought_sections, final_answer)
        """
        # Extract thought sections
        thought_pattern = r"Thought:|Search:|Evaluate:|Reasoning:"
        thought_matches = re.findall(f"({thought_pattern}.*?)(?={thought_pattern}|Answer:|$)", 
                                    full_response, re.DOTALL)
                                    
        thought_sections = [match.strip() for match in thought_matches]
        
        # Extract final answer
        answer_match = re.search(r"Answer:(.*?)$", full_response, re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else full_response
        
        return thought_sections, final_answer
    
    async def _validate_answer(
        self, 
        query: str, 
        response: str,
        context_docs: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Validate and potentially correct the answer against the context documents.
        
        Returns:
            Optional[str]: Corrected answer or None if validation failed
        """
        try:
            # Format context for validation
            context_text = self._format_context(context_docs)
            
            # Generate validation prompt
            prompt = CROSS_VALIDATION_TEMPLATE.format(
                context_docs=context_text,
                response=response,
                query=query
            )
            
            validation_response = await self.llm.agenerate([prompt])
            validation_text = validation_response.generations[0][0].text.strip()
            
            # Check if the response was marked as accurate
            if validation_text == "ACCURATE":
                return response
                
            # If not marked as accurate, return the corrected version
            if len(validation_text) > 10:  # Ensure we have a substantial response
                return validation_text
                
            # Default to original if validation unclear
            return response
            
        except Exception as e:
            logger.error(f"Error validating answer: {e}")
            return None
    
    async def _refine_arabic_text(self, query: str, response: str) -> str:
        """
        Refine Arabic text to sound more natural and human-like.
        
        Args:
            query: Original user query
            response: Generated response
            
        Returns:
            str: Refined Arabic text
        """
        try:
            from core.llm.prompt_templates import ARABIC_RESPONSE_VALIDATION_PROMPT
            
            # Skip if not Arabic or very short
            if not ArabicTextProcessor.contains_arabic(response) or len(response) < 20:
                return response
                
            prompt = ARABIC_RESPONSE_VALIDATION_PROMPT.format(
                query=query,
                response=response
            )
            
            refinement = await self.llm.agenerate([prompt])
            refined_text = refinement.generations[0][0].text.strip()
            
            # Only use the refinement if it's substantive
            if len(refined_text) > len(response) * 0.7:
                return refined_text
                
            # Default to original
            return response
            
        except Exception as e:
            logger.error(f"Error refining Arabic text: {e}")
            return response
