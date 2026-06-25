"""
Central configuration for the Redrob "Senior AI Engineer — Founding Team" ranker.

Everything role-specific lives here so the scoring is transparent and tunable in
one place. This mirrors the released job_description.docx as directly as possible:
the JD tells us its real must-haves, its explicit disqualifiers, and — in its note
to hackathon participants — that the right answer is *evidence of retrieval/ranking
work*, not the presence of AI keywords. We encode exactly that.

All term lists are matched case-insensitively against normalized text.
"""

# --------------------------------------------------------------------------- #
# Target role: title families
# --------------------------------------------------------------------------- #
# Titles that directly signal the target discipline (applied ML / IR / search).
RELEVANT_TITLE_TERMS = [
    "ml engineer", "machine learning", "ai engineer", "applied scientist",
    "applied ml", "nlp engineer", "data scientist", "search engineer",
    "recommendation systems", "recsys", "research engineer", "ai research",
    "ai specialist", "deep learning",
]

# Adjacent software titles — relevant *only if* they show retrieval/ranking
# evidence in their history; not relevant on the title alone.
ADJACENT_TITLE_TERMS = [
    "software engineer", "backend engineer", "data engineer", "analytics engineer",
    "full stack", "platform engineer", "staff engineer", "senior software",
]

# Non-technical (for this role) archetypes. A profile whose current title is one
# of these but whose skills are stuffed with AI terms is the classic stuffer trap.
NONTECH_TITLE_TERMS = [
    "hr manager", "marketing manager", "content writer", "graphic designer",
    "accountant", "sales executive", "customer support", "operations manager",
    "business analyst", "project manager", "civil engineer", "mechanical engineer",
    "qa engineer",  # quality-assurance, not ML
]

# Hard off-domain for this role (JD: CV/speech/robotics without NLP/IR).
OFFDOMAIN_TITLE_TERMS = ["computer vision", "cv engineer", "speech", "robotics"]

# --------------------------------------------------------------------------- #
# Domain evidence vocabulary — used to read *what work was actually done* from
# career-history descriptions, skills, and assessments (not just skill names).
# Grouped so we can credit breadth across sub-areas.
# --------------------------------------------------------------------------- #
EVIDENCE_TERMS = {
    "retrieval_ranking": [
        "retrieval", "ranking", "rank ", "learning to rank", "ltr", "search relevance",
        "semantic search", "information retrieval", "recommendation", "recommender",
        "recsys", "matching", "personalization",
    ],
    "embeddings": [
        "embedding", "embeddings", "sentence-transformer", "sentence transformer",
        "sbert", "bge", "e5", "word2vec", "vectorization", "dense vector", "encoder",
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
        "opensearch", "vector database", "vector db", "vector search", "ann index",
        "hybrid search", "bm25",
    ],
    "nlp": [
        "nlp", "natural language", "text classification", "named entity", "ner",
        "transformer", "bert", "llm", "language model", "question answering",
        "summarization", "rag", "fine-tun",
    ],
    "evaluation": [
        "ndcg", "mrr", "map@", "mean average precision", "precision@", "recall@",
        "a/b test", "ab test", "offline evaluation", "online evaluation", "eval framework",
    ],
    "ml_core": [
        "machine learning", "deep learning", "pytorch", "tensorflow", "scikit",
        "model training", "ml pipeline", "feature engineering", "mlops", "model serving",
    ],
}

# Must-have areas from the JD's "Things you absolutely need". Coverage across these
# is the backbone of the fit score. Maps each must-have to the evidence groups
# that satisfy it.
MUST_HAVES = {
    "embeddings_retrieval": ["embeddings", "retrieval_ranking"],
    "vector_db_hybrid_search": ["vector_db"],
    "ranking_evaluation": ["evaluation"],
    "applied_ml_depth": ["ml_core", "nlp"],
}

# Nice-to-haves (JD: "would like but won't reject for").
NICE_TO_HAVE_TERMS = [
    "lora", "qlora", "peft", "fine-tun", "xgboost", "lightgbm", "learning to rank",
    "hr-tech", "hrtech", "recruiting", "marketplace", "distributed systems",
    "large-scale inference", "open source", "open-source",
]

# --------------------------------------------------------------------------- #
# Disqualifiers (JD "Things we explicitly do NOT want" + the disqualifiers list)
# --------------------------------------------------------------------------- #
# Career spent only at IT-services / consulting firms (JD names these explicitly).
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree", "l&t infotech",
    "dxc", "mphasis", "hexaware", "igate", "syntel",
]

# Pure-research signal (academic/research-only, no production).
RESEARCH_ONLY_TERMS = [
    "research scholar", "phd researcher", "postdoc", "research assistant",
    "academic", "university", "institute of technology research",
]

# --------------------------------------------------------------------------- #
# Location preference (JD: Pune/Noida preferred; NCR/Hyd/Mumbai welcome; relocation)
# --------------------------------------------------------------------------- #
PREFERRED_LOCATIONS = ["noida", "pune"]
WELCOME_LOCATIONS = [
    "delhi", "new delhi", "gurgaon", "gurugram", "ghaziabad", "faridabad",
    "hyderabad", "mumbai", "navi mumbai", "bangalore", "bengaluru",
]

# --------------------------------------------------------------------------- #
# Scoring weights — additive base components (sum of weights = 1.0 before
# multiplicative modifiers). Documented & defensible for the Stage-5 interview.
# --------------------------------------------------------------------------- #
WEIGHTS = {
    "title_role_fit":      0.22,  # decisive anti-stuffer signal
    "domain_evidence":     0.24,  # outcome evidence, not vocabulary (catches Tier-5s)
    "must_have_coverage":  0.20,  # JD "absolutely need" list
    "semantic_similarity": 0.14,  # dense JD<->profile (hybrid retrieval)
    "experience_band":     0.08,  # peak 6-8y
    "skill_trust":         0.08,  # proficiency x duration x endorsement x assessment
    "nice_to_have":        0.04,  # bonus areas
}

# Multiplicative modifiers (applied after the additive base). Each in [floor, ceil].
MODIFIERS = {
    # Availability from redrob_signals: a great-on-paper but unreachable candidate
    # is "not actually available" (JD + signals doc) -> down-weight, don't zero.
    "availability_floor": 0.55,
    "availability_ceil": 1.10,
    # Location: preferred up-weight, relocation-willing keeps full credit.
    "location_pref": 1.08,
    "location_welcome": 1.03,
    "location_far_norelocate": 0.85,
    # Disqualifier multipliers (JD explicit). Soft, not hard-zero, except honeypots.
    "consulting_only": 0.45,
    "research_only": 0.45,
    "offdomain_only": 0.40,
    "title_hopper": 0.80,
    # Honeypots / impossible profiles -> forced to the bottom.
    "honeypot": 0.001,
    # Keyword stuffer (nontech title + unbacked AI skills) -> severe.
    "stuffer": 0.05,
}

# Experience band (years_of_experience). Peak credit in [PEAK_LO, PEAK_HI].
EXP_PEAK_LO, EXP_PEAK_HI = 6.0, 8.0
EXP_OK_LO, EXP_OK_HI = 4.0, 10.0

# Shortlist size from hybrid retrieval before detailed scoring.
SHORTLIST_SIZE = 2000

# Embedding model used offline (not imported at ranking time).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# "Today" anchor for recency math (dataset is synthetic; last_active maxes ~2026).
REFERENCE_DATE = "2026-06-01"
