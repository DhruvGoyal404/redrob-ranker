"""
Distilled JD retrieval query.

The released job_description.docx is long and conversational. For dense + sparse
retrieval we use a focused query that captures what the role *actually* needs
(the JD's own "Things you absolutely need" + the ideal-candidate paragraph),
deliberately excluding the chatty culture sections so retrieval keys on signal.

This text is the single document we embed at ranking time (one embedding) and the
query for BM25. Keeping it here makes the run fully reproducible and offline.
"""

JD_QUERY = (
    "Senior AI Engineer for a product company's intelligence layer. Owns ranking, "
    "retrieval and matching systems. Production experience with embeddings-based "
    "retrieval using sentence-transformers, BGE, E5 or OpenAI embeddings deployed to "
    "real users, handling embedding drift, index refresh and retrieval-quality "
    "regression. Production experience with vector databases or hybrid search: "
    "Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch or Elasticsearch. Strong "
    "Python and code quality. Designs evaluation frameworks for ranking systems: "
    "NDCG, MRR, MAP, offline-to-online correlation, A/B testing. Built and shipped an "
    "end-to-end search, ranking or recommendation system at meaningful scale at a "
    "product company. Applied machine learning and NLP, hybrid retrieval, dense and "
    "sparse search, learning to rank, LLM fine-tuning with LoRA or QLoRA. Six to eight "
    "years experience, mostly applied ML at product companies rather than services or "
    "pure research. Based in or willing to relocate to Noida or Pune."
)

# Short keyword set used for BM25 emphasis / quick title hints.
JD_KEYWORDS = [
    "embeddings", "retrieval", "ranking", "vector", "search", "nlp", "recommendation",
    "ndcg", "evaluation", "python", "applied machine learning", "hybrid search",
]
