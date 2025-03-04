import asyncio
import pytest
from core.search.reranker import QueryResultReranker

@pytest.fixture
def mock_results():
    """Sample search results for testing"""
    return [
        {
            "content": "Ahmed is the CEO of XYZ company. He has 15 years of experience in the industry.",
            "metadata": {"title": "About Ahmed", "source": "company_bio.pdf"}
        },
        {
            "content": "The company was founded in 2005 by Ahmed and Sarah. It specializes in AI solutions.",
            "metadata": {"title": "Company History", "source": "history.pdf"}
        },
        {
            "content": "Machine learning algorithms are used to predict market trends.",
            "metadata": {"title": "Technology", "source": "tech.pdf"}
        },
        {
            "content": "أحمد هو الرئيس التنفيذي لشركة XYZ. لديه خبرة 15 عاما في الصناعة.",
            "metadata": {"title": "عن أحمد", "source": "company_bio_ar.pdf"}
        }
    ]

@pytest.mark.asyncio
async def test_keyword_rerank():
    """Test keyword reranking functionality"""
    reranker = QueryResultReranker()
    results = mock_results()
    
    # English query
    reranked = await reranker._keyword_rerank("Who is Ahmed?", results)
    
    # Check that reranking was applied
    assert all("relevance_score" in item for item in reranked)
    
    # Check sorting order
    assert reranked[0]["metadata"]["title"] in ["About Ahmed", "عن أحمد"]
    
    # Arabic query
    reranked_ar = await reranker._keyword_rerank("من هو أحمد؟", results)
    
    # Check that Arabic document is prioritized for Arabic query
    assert reranked_ar[0]["metadata"]["title"] == "عن أحمد"

@pytest.mark.asyncio
async def test_rerank_results():
    """Test the main reranking method"""
    reranker = QueryResultReranker()
    results = mock_results()
    
    reranked = await reranker.rerank_results("Ahmed CEO", results, top_k=2)
    
    # Check that we got the right number of results
    assert len(reranked) == 2
    
    # The first result should be about Ahmed
    assert "Ahmed" in reranked[0]["content"]
    assert "CEO" in reranked[0]["content"]

if __name__ == "__main__":
    asyncio.run(test_keyword_rerank())
    asyncio.run(test_rerank_results())
    print("All tests completed successfully!")
