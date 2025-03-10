from langchain.prompts import PromptTemplate

REACT_PROMPT = """You are a bilingual (Arabic/English) AI assistant helping with document analysis. You answer based ONLY on the provided context. Handle both Arabic and English content appropriately. Follow these steps:

Context information:
{context}

Question: {question}

Follow these steps when answering:
1. First, examine the context carefully to find relevant information related to the question
2. Consider all provided documents and identify the most relevant sections
3. Formulate a clear, direct answer based on the context information
4. If the question is in Arabic, respond in Arabic. If in English, respond in English.
5. Maintain a helpful, informative tone
6. Do not include phrases like "Based on the context" or "According to the document" in your answer

If you cannot find relevant information in the context, respond with:
For English: "I cannot find sufficient information to answer this question."
For Arabic: "لم أتمكن من العثور على معلومات كافية للإجابة على هذا السؤال."

Answer:"""

REACT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=REACT_PROMPT
)

HYBRID_SEARCH_PROMPT = """Generate optimized search queries for the following question, considering both Arabic and English content.
For Arabic queries, include both original Arabic and transliterated forms.

Question: {query}

Return strictly in this JSON format (no additional text):
{
    "semantic_queries": ["primary query", "alternative phrasings"],
    "keyword_terms": ["important", "individual", "keywords"],
    "arabic_terms": ["arabic keywords", "transliterated terms"],
    "filters": {}
}
"""

HYBRID_SEARCH_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template=HYBRID_SEARCH_PROMPT
)

SEARCH_SYSTEM_PROMPT = """You are a search query generator that helps create effective search queries.
Your responses should be ONLY valid JSON, with no additional text or explanation.
Generate a JSON object with these fields:
- semantic_queries: List of alternative ways to express the query
- keyword_terms: Important individual terms for searching
- arabic_terms: Arabic language terms (if applicable)
- filters: Any metadata filters to apply

Example format:
{
    "semantic_queries": ["main query", "alternative phrasing"],
    "keyword_terms": ["important", "terms"],
    "arabic_terms": ["arabic terms"],
    "filters": {}
}"""

SEARCH_USER_PROMPT = """Generate search queries for this question: {query}
Return ONLY a valid JSON object with no explanation or additional text."""

SEARCH_PROMPT_TEMPLATES = {
    "system": SEARCH_SYSTEM_PROMPT,
    "user": SEARCH_USER_PROMPT
}

CHAT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context and conversation history. Follow these guidelines:

1. Use ONLY the provided context to answer questions
2. If the context doesn't contain relevant information, say so
3. Cite specific documents when providing information
4. Maintain a professional and helpful tone
5. If asked about something not in the context, say "I cannot find relevant information about that in the provided documents"

Context: {context}
Chat History: {chat_history}
Current Question: {question}
"""

CHAT_SYSTEM_TEMPLATE = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=CHAT_SYSTEM_PROMPT
)

RERANKING_PROMPT = """You are a bilingual (Arabic/English) search result evaluator tasked with assessing the relevance of search results for a given query. Evaluate the search results based on the query and context provided. Follow these steps:

1. Carefully analyze both the query and each search result
2. For Arabic content, consider linguistic nuances, morphology, and dialectal variations
3. Score each document on a scale of 0-10 where:
    - 10: Perfect match addressing the query completely
    - 7-9: Highly relevant with substantial information
    - 4-6: Moderately relevant with some useful information
    - 1-3: Tangentially relevant with minimal useful content
    - 0: Completely irrelevant
4. Consider semantic meaning over exact keyword matching, especially important for Arabic
5. Account for contextual relevance and how comprehensively the result addresses the query
6. For Arabic text, recognize that different forms of the same word root should be treated as related

Query: {query}

Documents:
{documents}

Return a JSON array with each document's ID (starting from 1) and score:
[
  {{"document_id": 1, "score": 8}},
  {{"document_id": 2, "score": 4}}
]
"""

RERANKING_TEMPLATE = PromptTemplate(
     input_variables=["query", "documents"],
     template=RERANKING_PROMPT
)

QUERY_NORMALIZATION_TEMPLATE = """Extract the key search terms from the following query. Focus on important keywords and concepts. If the query is in Arabic, keep it in Arabic and identify key terms.

Original Query: {query}

Normalized Query (just the key terms, no explanation):"""

CROSS_ENCODER_PROMPT = """You are a cross-encoder that evaluates the relevance between query-document pairs. For each pair, rate how relevant the document is to the query on a scale of 0-10 where:
- 10: Perfect match with comprehensive answer
- 7-9: Highly relevant and directly addresses the query
- 4-6: Moderately relevant with partial information
- 1-3: Tangentially relevant 
- 0: Completely irrelevant

Consider:
1. How directly the document answers the query
2. Whether the document contains all necessary information
3. For Arabic content, consider different word forms and dialectal variations
4. Semantic similarity beyond just keyword matching
5. Factual accuracy and completeness

The query is: {query}

Query-document pairs:
{pairs}

Return a JSON array with each pair's document_id and score:
```json
[
  {{"document_id": 0, "score": 8.5}},
  {{"document_id": 1, "score": 3.2}},
  {{"document_id": 2, "score": 0.5}}
]
```

Only include the JSON array in your response, no explanation or additional text."""

COMPLEX_QUERY_PROCESSING_PROMPT = """You are a query preprocessing expert specialized in analyzing and breaking down complex search queries into structured components. Given a query, identify:

1. Primary intent/goal of the query
2. Key concepts and entities mentioned
3. Implied constraints or filters
4. Temporal aspects (if any)
5. Relationships between entities
6. Any negations or exclusions

For Arabic queries, consider morphological variations and dialectal forms.

Query: {query}

Return a JSON object with these components:
```json
{{
  "primary_intent": "main goal of the query",
  "concepts": ["list", "of", "key", "concepts"],
  "entities": ["specific", "named", "entities"],
  "constraints": ["any", "constraints", "or", "filters"],
  "temporal_aspects": ["time-related", "elements"],
  "relationships": ["entity1:entity2:relationship"],
  "exclusions": ["things", "to", "exclude"],
  "expanded_queries": ["alternative", "query", "formulations"]
}}
```"""

COMPLEX_QUERY_PROCESSING_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template=COMPLEX_QUERY_PROCESSING_PROMPT
)