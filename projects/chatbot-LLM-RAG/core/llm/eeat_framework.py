"""
EEAT Framework Module
======================
Defines the Experience, Expertise, Authority, Trust framework
used throughout the content creation pipeline.

Provides scoring rubrics, validation rules, and signal definitions.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EEATSignal:
    """A single EEAT signal with its metadata."""
    name: str
    description: str
    content_markers: list[str] = field(default_factory=list)
    required: bool = True


@dataclass
class EEATScore:
    """EEAT evaluation score for an article."""
    experience: float = 0.0  # 0-25
    expertise: float = 0.0   # 0-25
    authority: float = 0.0   # 0-25
    trust: float = 0.0       # 0-25

    @property
    def total(self) -> float:
        return self.experience + self.expertise + self.authority + self.trust

    @property
    def grade(self) -> str:
        total = self.total
        if total >= 80:
            return "A"
        elif total >= 60:
            return "B"
        elif total >= 40:
            return "C"
        elif total >= 20:
            return "D"
        return "F"

    def to_dict(self) -> dict:
        return {
            "experience": self.experience,
            "expertise": self.expertise,
            "authority": self.authority,
            "trust": self.trust,
            "total": self.total,
            "grade": self.grade,
        }


class EEATFramework:
    """
    EEAT Framework - Defines and validates EEAT signals in content.

    Experience → Practical examples, real-world usage, case studies
    Expertise  → Technical depth, data, methodology explanations
    Authority  → Structured comparisons, citations, cross-references
    Trust      → Transparency, clear positioning, honest conclusions
    """

    SIGNALS = {
        "experience": EEATSignal(
            name="Experience",
            description="Practical examples and real-world scenarios with strategic analysis",
            content_markers=[
                "case study",
                "real-world example",
                "in practice",
                "hands-on",
                "we tested",
                "our experience",
                "step-by-step",
                "when using",
                "in our setup",
                "practical tip",
                "strategic move",
                "execution model",
                "why it worked",
                "strategic takeaway",
            ],
        ),
        "expertise": EEATSignal(
            name="Expertise",
            description="Technical explanations, data tables, methodology",
            content_markers=[
                "technical breakdown",
                "how it works",
                "architecture",
                "protocol",
                "algorithm",
                "benchmark",
                "performance data",
                "specification",
                "configuration",
                "implementation",
            ],
        ),
        "authority": EEATSignal(
            name="Authority",
            description="Structured comparisons, citations, authority signals",
            content_markers=[
                "comparison table",
                "vs",
                "compared to",
                "according to",
                "research shows",
                "data from",
                "industry standard",
                "best practice",
                "expert recommendation",
                "structured comparison",
            ],
        ),
        "trust": EEATSignal(
            name="Trust",
            description="Transparency, clear positioning, reliability signals",
            content_markers=[
                "honest assessment",
                "limitation",
                "disclaimer",
                "important to note",
                "keep in mind",
                "transparency",
                "we recommend",
                "our verdict",
                "pros and cons",
                "fair evaluation",
            ],
        ),
    }

    # Section templates that fulfill EEAT requirements
    SECTION_TEMPLATES = {
        "experience": {
            "heading_patterns": [
                "Real-World {topic} in Action",
                "Practical Guide to {topic}",
                "How We Use {topic} (Case Study)",
                "{topic}: Hands-On Experience",
            ],
            "required_elements": [
                "At least one strategic case study (Context → Strategic Move → Execution → Why It Worked → Takeaway)",
                "Step-by-step practical guidance",
                "Lessons learned or tips from experience",
                "Case studies must use qualitative outcomes — NO fabricated numbers or unverified brand names",
            ],
        },
        "expertise": {
            "heading_patterns": [
                "Technical Breakdown: How {topic} Works",
                "The Technology Behind {topic}",
                "{topic} Architecture Explained",
                "Deep Dive: {topic} Performance Data",
            ],
            "required_elements": [
                "Technical explanation with data",
                "Data table or performance metrics",
                "Methodology or protocol explanation",
            ],
        },
        "authority": {
            "heading_patterns": [
                "{topic} vs Alternatives: Structured Comparison",
                "How {topic} Stacks Up Against Competitors",
                "Industry Analysis: {topic} Landscape",
            ],
            "required_elements": [
                "Comparison table with clear criteria",
                "Reference to industry data or standards",
                "Cross-reference to authoritative sources",
            ],
        },
        "trust": {
            "heading_patterns": [
                "Our Honest Assessment of {topic}",
                "{topic}: Pros, Cons, and Verdict",
                "What You Need to Know Before Choosing {topic}",
            ],
            "required_elements": [
                "Balanced pros and cons",
                "Clear disclosure and positioning",
                "Transparent recommendations with reasoning",
            ],
        },
    }

    @classmethod
    def score_content(cls, content: str) -> EEATScore:
        """Score content against EEAT markers."""
        content_lower = content.lower()
        score = EEATScore()

        for signal_key, signal in cls.SIGNALS.items():
            hits = sum(1 for marker in signal.content_markers if marker in content_lower)
            # Max 25 points per signal, 2.5 per marker hit
            signal_score = min(hits * 2.5, 25.0)
            setattr(score, signal_key, signal_score)

        return score

    @classmethod
    def get_missing_signals(cls, content: str) -> list[dict]:
        """Identify which EEAT signals are weak or missing in content."""
        score = cls.score_content(content)
        missing = []

        thresholds = {"experience": 10, "expertise": 10, "authority": 10, "trust": 10}

        for key, threshold in thresholds.items():
            current = getattr(score, key)
            if current < threshold:
                signal = cls.SIGNALS[key]
                missing.append({
                    "signal": key,
                    "name": signal.name,
                    "current_score": current,
                    "threshold": threshold,
                    "description": signal.description,
                    "suggested_markers": signal.content_markers[:5],
                })

        return missing

    @classmethod
    def get_section_template(cls, signal: str, topic: str) -> dict:
        """Get a section template for a specific EEAT signal."""
        template = cls.SECTION_TEMPLATES.get(signal, {})
        if template:
            return {
                "headings": [h.format(topic=topic) for h in template.get("heading_patterns", [])],
                "required_elements": template.get("required_elements", []),
            }
        return {"headings": [], "required_elements": []}
