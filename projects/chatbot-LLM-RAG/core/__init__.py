"""
SEO Bot Core Engine
====================
Modular architecture with 8 core engines:

core/
├── database.py              # ORM Engine + Session
├── models.py                # SQLAlchemy models
├── llm/                     # LLM interaction layer
├── serp/                    # SERP Intelligence Engine
├── intent/                  # Intent Mapping & Gap Detection
├── cluster/                 # Topic Cluster Architect
├── content/                 # Content generation (Brand, EEAT, Outline)
├── linking/                 # Internal Link Automation
├── monitoring/              # Rank Monitor + Update Engine
├── quality/                 # Quality Gate (Human-in-the-loop)
└── analytics/               # GSC Performance Analyzer
"""

__version__ = "1.0.0"
