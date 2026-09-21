from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_community.document_compressors import FlashrankRerank


def build_advanced_retriever(documents,vector_retriever):
    """ Combines MB25 and vector search, then applies FlashRank re-ranking """

    # 1. BM25 Keyword retriever (Great for exact ingredient names)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    # 2. Hybrid Retriever (keyword+Semantic vector)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever,vector_retriever],
        weights=[0.5,0.5]
    )

    # 3. Re-ranker (search the results user's questions)

    compressor= FlashrankRerank(top_n=3)
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )
    
    return advanced_retriever
