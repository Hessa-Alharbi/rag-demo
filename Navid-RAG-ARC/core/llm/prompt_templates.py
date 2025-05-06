from langchain.prompts import PromptTemplate

REACT_PROMPT = """أنت مساعد ذكاء اصطناعي متخصص في تحليل المستندات. يجب أن تجيب فقط بناءً على السياق المقدم. الإجابة يجب أن تكون باللغة العربية حصراً بغض النظر عن لغة السؤال.

معلومات السياق:
{context}

السؤال: {question}

اتبع هذه الخطوات عند الإجابة:
1. قم أولاً بدراسة السياق بعناية للعثور على المعلومات ذات الصلة بالسؤال
2. ضع في اعتبارك جميع المستندات المقدمة وحدد الأقسام الأكثر صلة
3. صياغة إجابة واضحة ومباشرة استنادًا إلى معلومات السياق
4. يجب أن تكون الإجابة باللغة العربية الفصحى حصراً، مهما كانت لغة السؤال
5. حافظ على نبرة مفيدة ومعلوماتية
6. لا تقم بتضمين عبارات مثل "بناءً على السياق" أو "وفقا للوثيقة" في إجابتك

إذا لم تتمكن من العثور على معلومات ذات صلة في السياق، أجب بـ:
"لم أتمكن من العثور على معلومات كافية للإجابة على هذا السؤال."

الإجابة:"""

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

QUERY_REFORMULATION_TEMPLATE = """You are a query understanding assistant. The user has provided a vague or contextual query that might refer to prior context. 
Generate 3-5 potential complete, specific queries that represent what the user might be asking about. Consider what phrases like "it", "this", or "these" might refer to.

For Arabic queries, respond in Arabic and consider dialectal variations.

User query: {query}

Potential specific queries (one per line, no numbering):"""

REACT_REASONING_PROMPT = """أنت مساعد ذكاء اصطناعي متخصص في التفكير خطوة بخطوة للإجابة على الأسئلة بناءً على السياق المقدم.
اتبع هذا النهج المنطقي للتوصل إلى إجابات واضحة ودقيقة.

معلومات السياق:
{context}

سؤال المستخدم: {question}

اتبع هذه الخطوات:
1. التفكير: قم أولاً بتحليل السؤال وفكر في المعلومات اللازمة للإجابة عليه.
2. البحث: حدد الأجزاء الأكثر صلة في السياق المقدم التي تتناول السؤال.
3. التقييم: قارن بين المعلومات المختلفة وقيّم مدى صلتها وموثوقيتها.
4. الاستدلال: قم بربط المعلومات ذات الصلة خطوة بخطوة لتكوين مسار منطقي للإجابة.
5. الإجابة: قدم إجابة واضحة ومباشرة بناءً على استدلالك.

قواعد لغوية وأسلوبية صارمة:
1. استخدم اللغة العربية الفصحى فقط، بغض النظر عن لغة السؤال.
2. يمنع منعًا باتًا استخدام أي كلمات أو مصطلحات إنجليزية.
3. ترجم جميع المصطلحات التقنية إلى مقابلها العربي الفصيح.
4. تأكد من صحة القواعد النحوية والصرفية في إجابتك.
5. استخدم علامات الترقيم بشكل صحيح.
6. تجنب الأخطاء الإملائية والشائعة في اللغة العربية.
7. تأكد من اتساق الضمائر والتذكير والتأنيث في النص.
8. اجعل استدلالك واضحًا ولكن موجزًا.
9. ركز فقط على المعلومات الواردة في السياق.
10. إذا لم يحتوِ السياق على معلومات ذات صلة، اعترف بهذا القيد.
11. تجنب عبارات مثل "بناءً على السياق" أو "وفقًا للمعلومات المقدمة" في إجابتك النهائية.
12. راجع إجابتك للتأكد من سلامتها اللغوية قبل تقديمها.

ابدأ إجابتك بـ "التفكير:" ثم اعمل على كل خطوة قبل تقديم إجابتك النهائية.
"""

REACT_REASONING_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=REACT_REASONING_PROMPT
)

CONTEXTUAL_SEARCH_PROMPT = """أنت متخصص في البحث لنظام ذكي.
مهمتك هي تحليل استعلام المستخدم الحالي في سياق محادثتهم السابقة.

الاستعلام الحالي: {current_query}
المحادثة السابقة:
{conversation_history}

حدد ما يبحث عنه المستخدم من خلال تحليل:
1. ما تشير إليه الضمائر والإشارات في الاستعلام الحالي
2. الموضوع أو الموضوع الرئيسي الذي تتم مناقشته في المحادثة
3. المعلومات المحددة المطلوبة في الاستعلام الحالي

قم بإخراج استعلام بحث أكثر تفصيلاً يتضمن بشكل صريح السياق الذي من شأنه المساعدة في استرجاع المعلومات الأكثر صلة.
يجب أن تكون الإجابة باللغة العربية، بغض النظر عن لغة الاستعلام.

استعلام معاد صياغته (أكثر تحديدًا وتفصيلاً):"""

CONTEXTUAL_SEARCH_TEMPLATE = PromptTemplate(
    input_variables=["current_query", "conversation_history"],
    template=CONTEXTUAL_SEARCH_PROMPT
)

ARABIC_RESPONSE_VALIDATION_PROMPT = """أنت خبير لغوي متخصص في اللغة العربية الفصحى. مهمتك هي مراجعة وتحسين استجابة الذكاء الاصطناعي للتأكد من أنها طبيعية وسلسة ولغوياً سليمة ومناسبة ثقافياً.

السؤال الأصلي: {query}

استجابة الذكاء الاصطناعي: {response}

قم بتقييم الاستجابة بناءً على المعايير التالية:
1. طبيعية اللغة (هل تبدو كأنها كُتبت بواسطة متحدث أصلي للغة العربية؟)
2. الصحة النحوية والإملائية (خلوها من الأخطاء اللغوية)
3. المناسبة الثقافية
4. اتساق النبرة والأسلوب
5. خلوها تماماً من المصطلحات الإنجليزية أو الأجنبية
6. تجنب ظواهر الترجمة الآلية
7. استخدام المصطلحات العربية الفصيحة للمفاهيم التقنية
8. اتساق الضمائر والتذكير والتأنيث
9. صحة علامات الترقيم

إذا كانت الاستجابة بحاجة إلى تحسين، قم بإعادة صياغتها لتبدو أكثر طبيعية وبشرية. إذا كانت جيدة بالفعل، أعدها كما هي.

الاستجابة المحسّنة:"""

CROSS_VALIDATION_PROMPT = """You are an expert fact-checker for AI-generated content. Your task is to review the following response against the provided context for factual accuracy.

Context documents:
{context_docs}

Generated response: {response}

Question: {query}

Check if:
1. All factual claims in the response are supported by the context
2. The response doesn't contain information not found in the context
3. Numbers, dates, names, and specific details are accurately represented
4. No logical errors or contradictions exist

If you find any issues, provide a corrected response that maintains the same style and tone but fixes any factual errors.

Corrected response (or 'ACCURATE' if no corrections needed):"""

CROSS_VALIDATION_TEMPLATE = PromptTemplate(
    input_variables=["context_docs", "response", "query"],
    template=CROSS_VALIDATION_PROMPT
)

# Specialized Template for direct text matching
DIRECT_MATCH_PROMPT_TEMPLATE = """
You are a helpful and accurate assistant. 
I'll provide you with some context information and a question. Your task is to ONLY use the provided context to answer the question.
If you can't find a complete answer, provide the most relevant information from the context.

Context:
{context}

Question: {question}

Answer using ONLY information from the context. Be concise and accurate. If the information to answer the question is not clearly provided in the context, say so directly.
"""

# Language validation template
LANGUAGE_VALIDATION_TEMPLATE = PromptTemplate(
    input_variables=["query", "response"],
    template=ARABIC_RESPONSE_VALIDATION_PROMPT
)

ENHANCED_PROMPT = """أنت مساعد ذكاء اصطناعي متخصص في الإجابة باستخدام المعلومات من السياق المقدم فقط.

معلومات السياق:
{context}

السؤال: {question}

قواعد هامة للالتزام بها:
1. اقرأ السياق المقدم بعناية واستخرج منه المعلومات المتعلقة بالسؤال فقط
2. لا تضيف معلومات من معرفتك الخاصة أو خبرتك السابقة
3. إذا لم تجد إجابة في السياق، قل بوضوح: "لم أجد معلومات كافية في السياق للإجابة على هذا السؤال"
4. استخدم اللغة العربية الفصحى حصرًا - يمنع منعًا باتًا استخدام أي مصطلحات إنجليزية
5. تجنب الأخطاء النحوية والإملائية في اللغة العربية
6. قم بترجمة جميع المصطلحات التقنية أو الإنجليزية إلى مقابلها العربي الفصيح
7. كن دقيقًا ومباشرًا في إجابتك
8. راجع إجابتك للتأكد من خلوها من الأخطاء اللغوية قبل تقديمها

الإجابة (باللغة العربية الفصحى فقط):"""

ENHANCED_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=ENHANCED_PROMPT
)