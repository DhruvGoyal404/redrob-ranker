"""
Example job descriptions for the live demo's role selector.

The demo's ranker (app/demo_rank.py) is genuinely JD-adaptive - relevance to the JD
(dense semantic + BM25) drives the ranking - so switching between these contrasting
roles visibly re-ranks the candidates. The first is the challenge's target role.
"""
from src.jd import JD_QUERY

PRESETS = {
    "Senior AI Engineer (challenge role)": JD_QUERY,

    "Computer Vision Engineer": (
        "Computer Vision Engineer for a robotics product. Production experience with image "
        "classification, object detection and segmentation using PyTorch or TensorFlow, CNNs, "
        "ViTs, and real-time inference on edge devices. Camera calibration, 3D geometry, video "
        "analytics, and model optimization with TensorRT or ONNX. Five plus years shipping "
        "vision models to real users."
    ),

    "Backend / Distributed Systems Engineer": (
        "Senior Backend Engineer for a high-scale platform. Designs and operates distributed "
        "systems and microservices in Go or Java, with Kafka, gRPC, PostgreSQL and Redis. Strong "
        "on API design, concurrency, observability, horizontal scaling, fault tolerance and "
        "on-call ownership. Kubernetes and cloud infrastructure on AWS or GCP. Six plus years "
        "building reliable back-end services at product companies."
    ),

    "Data Analyst (Business Intelligence)": (
        "Data Analyst for a business-intelligence team. Strong SQL, dashboarding in Tableau or "
        "Power BI, experiment analysis and stakeholder reporting. Turns messy data into clear "
        "business recommendations, builds metrics and funnels, and partners with product and "
        "marketing. Python or R for analysis is a plus. Three to six years in analytics."
    ),
}

DEFAULT_PRESET = "Senior AI Engineer (challenge role)"
