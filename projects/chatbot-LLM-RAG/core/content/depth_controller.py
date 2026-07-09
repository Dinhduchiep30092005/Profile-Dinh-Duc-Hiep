"""
Depth Controller — Adjusts content depth strategy per section based on search intent.

Solves the problem of all sections having the same depth/pattern/tone.
Each intent type gets a different writing strategy:

    informational → explain + real example + data reference
    commercial    → problem → solution → product bridge + comparison table
    comparison    → contrast → table → recommendation + counter-arguments
    transactional → benefit → proof → clear CTA

Also controls:
    - Section rhythm (varying paragraph lengths, alternating analysis/example)
    - Tone variation across sections (some analytical, some narrative, some data-heavy)
    - Mandatory elements per section type
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# Section tone profiles for rhythm variation
# ──────────────────────────────────────────

_TONE_PROFILES = [
    {
        "name": "analytical",
        "instruction": (
            "Write in an ANALYTICAL tone. Lead with data, statistics, or structured reasoning. "
            "Use precise numbers and percentages. Structure arguments as: claim → evidence → implication. "
            "Prefer tables or bullet comparisons over long paragraphs."
        ),
        "paragraph_style": "short_precise",  # 2-3 sentence paragraphs
    },
    {
        "name": "narrative",
        "instruction": (
            "Write in a NARRATIVE tone. Lead with a concrete scenario or mini case study. "
            "Describe a real situation step by step, then extract the lesson. "
            "Use storytelling: set the scene → describe the challenge → show the resolution → state the insight."
        ),
        "paragraph_style": "flowing",  # 3-4 sentence paragraphs
    },
    {
        "name": "technical",
        "instruction": (
            "Write in a TECHNICAL tone. Explain the underlying mechanism or process in detail. "
            "Use correct terminology with brief inline definitions. "
            "Structure as: what it is → how it works → why it matters → edge cases or limitations."
        ),
        "paragraph_style": "structured",  # mixed lists + paragraphs
    },
    {
        "name": "contrarian",
        "instruction": (
            "Write with a CONTRARIAN perspective. Start by stating the common belief, then challenge it "
            "with specific evidence or alternative viewpoints. Structure as: "
            "common assumption → why it's incomplete/wrong → the more nuanced reality → practical implication."
        ),
        "paragraph_style": "argumentative",
    },
    {
        "name": "data_driven",
        "instruction": (
            "Write in a DATA-DRIVEN tone. Every major claim must reference a specific number, "
            "percentage, timeframe, or measurable outcome. Structure as: "
            "metric/stat → what it means → practical significance → actionable implication. "
            "Include at least one mini comparison or data table."
        ),
        "paragraph_style": "short_precise",
    },
]


# ──────────────────────────────────────────
# Language-aware layer labels
# ──────────────────────────────────────────

_LAYER_LABELS = {
    "en": {
        "L1": "CONTEXT FRAMING",
        "L2": "CORE EXPLANATION",
        "L3": "PRACTICAL SCENARIO",
        "L4": "STRATEGIC TAKEAWAY",
        "problem": "PROBLEM FRAMING",
        "decision": "DECISION CONTEXT",
        "value": "VALUE PROPOSITION",
        "context": "CONTEXT",
        "experience": "Experience-based",
        "expertise": "Technical",
        "authority": "Data-backed",
        "trust": "Balanced",
        "recommendation": "RECOMMENDATION",
        "next_step": "NEXT STEP",
        "decision_guidance": "DECISION GUIDANCE",
        "takeaway": "TAKEAWAY",
    },
    "vi": {
        "L1": "BỐI CẢNH MỞ ĐẦU",
        "L2": "PHÂN TÍCH CỐT LÕI",
        "L3": "TÌNH HUỐNG THỰC TẾ",
        "L4": "ĐIỂM NHẤN HÀNH ĐỘNG",
        "problem": "XÁC ĐỊNH VẤN ĐỀ",
        "decision": "BỐI CẢNH QUYẾT ĐỊNH",
        "value": "GIÁ TRỊ CỐT LÕI",
        "context": "BỐI CẢNH",
        "experience": "Dựa trên trải nghiệm",
        "expertise": "Chuyên sâu kỹ thuật",
        "authority": "Dựa trên dữ liệu",
        "trust": "Cân bằng đa chiều",
        "recommendation": "KHUYẾN NGHỊ",
        "next_step": "BƯỚC TIẾP THEO",
        "decision_guidance": "HƯỚNG DẪN LỰA CHỌN",
        "takeaway": "ĐIỂM THEN CHỐT",
    },
}


def _get_labels(language: str) -> dict:
    """Get layer labels for the given language, fallback to English."""
    lang_key = "vi" if language and language.lower().startswith("vi") else "en"
    return _LAYER_LABELS.get(lang_key, _LAYER_LABELS["en"])


class DepthController:
    """Controls content depth and writing strategy based on search intent and section position."""

    def __init__(self):
        self._tone_index = 0  # Cycles through tone profiles for variety

    def get_section_strategy(
        self,
        intent_type: str,
        section: dict,
        section_index: int,
        total_sections: int,
        children: list[dict] | None = None,
        language: str = "vi",
    ) -> dict:
        """Get a tailored depth strategy for a specific section.

        Args:
            intent_type: Search intent (informational/commercial/comparison/transactional)
            section: The H2 section dict
            section_index: Position of this H2 (0-based)
            total_sections: Total number of H2 sections
            children: List of H3/H4 children

        Returns:
            dict with:
                - depth_instruction: Detailed writing strategy for this section
                - four_layer_structure: Structured 4-layer requirement
                - tone_profile: Which tone to use
                - mandatory_elements: List of required content elements
                - anti_patterns: Things to explicitly avoid
        """
        content_type = section.get("content_type", "text").lower()
        eeat_signal = section.get("eeat_signal", "expertise").lower()

        # ── 1. Select intent-based depth template ──
        intent_depth = self._get_intent_depth(intent_type, content_type)

        # ── 2. Build 4-layer structure requirement (language-aware) ──
        four_layer = self._build_four_layer(intent_type, content_type, eeat_signal, language)

        # ── 3. Select tone profile (cycling for rhythm) ──
        tone = self._select_tone(section_index, total_sections, content_type, eeat_signal)

        # ── 4. Determine mandatory elements ──
        mandatory = self._get_mandatory_elements(intent_type, content_type, eeat_signal)

        # ── 5. Anti-patterns specific to this section ──
        anti_patterns = self._get_anti_patterns(section_index, total_sections)

        strategy = {
            "depth_instruction": intent_depth,
            "four_layer_structure": four_layer,
            "tone_profile": tone,
            "mandatory_elements": mandatory,
            "anti_patterns": anti_patterns,
        }

        logger.debug(
            f"[DepthController] Section {section_index + 1}/{total_sections}: "
            f"intent={intent_type}, tone={tone['name']}, "
            f"mandatory={len(mandatory)} elements"
        )

        return strategy

    # ──────────────────────────────────────────
    # Intent-based depth
    # ──────────────────────────────────────────

    def _get_intent_depth(self, intent_type: str, content_type: str) -> str:
        """Get depth instruction based on search intent."""

        if intent_type == "informational":
            return (
                "DEPTH STRATEGY (Informational Intent):\n"
                "The reader wants to UNDERSTAND. Your job is to teach, not sell.\n"
                "1. START with a concrete fact, statistic, or specific observation — NOT a definition\n"
                "2. EXPLAIN the mechanism: how does this actually work? What's the process?\n"
                "3. ILLUSTRATE with a real scenario: 'For example, a [specific type of user] "
                "doing [specific action] would see [specific result]...'\n"
                "4. QUANTIFY wherever possible: timeframes, percentages, metrics, thresholds\n"
                "5. END with a strategic takeaway the reader can apply immediately\n\n"
                "AVOID: Generic definitions, obvious statements, 'It's important to note that...'"
            )

        elif intent_type == "commercial":
            return (
                "DEPTH STRATEGY (Commercial Intent):\n"
                "The reader is evaluating options. Help them decide with evidence.\n"
                "1. START by naming the specific PROBLEM the reader faces\n"
                "2. ANALYZE the solution landscape: what approaches exist? What are trade-offs?\n"
                "3. COMPARE with specific criteria (not vague 'pros and cons'): "
                "cost, time-to-result, scalability, risk level\n"
                "4. BRIDGE to the recommended solution with concrete evidence: "
                "'Based on [metric], approach X delivers [specific result] in [timeframe]'\n"
                "5. END with a clear recommendation backed by reasoning\n\n"
                "AVOID: Premature selling, generic benefits lists, unsupported superlatives"
            )

        elif intent_type == "comparison":
            return (
                "DEPTH STRATEGY (Comparison Intent):\n"
                "The reader is comparing specific options. Be the honest broker.\n"
                "1. START with the decision criteria that actually matter\n"
                "2. CONTRAST options using a structured framework (table preferred)\n"
                "3. ACKNOWLEDGE trade-offs: every option has downsides — name them\n"
                "4. RECOMMEND based on user scenario: 'If you need X, choose A. If Y matters more, B.'\n"
                "5. END with a nuanced recommendation, not a blanket 'best' pick\n\n"
                "AVOID: One-sided analysis, missing counter-arguments, vague 'depends on your needs'"
            )

        elif intent_type == "transactional":
            return (
                "DEPTH STRATEGY (Transactional Intent):\n"
                "The reader is ready to act. Remove friction and build confidence.\n"
                "1. START with the key BENEFIT or value proposition — what they get\n"
                "2. PROVE it works: specific results, case outcomes, data points\n"
                "3. ADDRESS objections proactively: cost, complexity, risks + how they're mitigated\n"
                "4. SHOW the process: what happens after they take action (step by step)\n"
                "5. END with clear next step and urgency (without being pushy)\n\n"
                "AVOID: Feature dumps without benefits, ignoring concerns, aggressive sales language"
            )

        else:
            return (
                "DEPTH STRATEGY:\n"
                "1. Lead with the most specific, actionable insight\n"
                "2. Support with evidence, data, or concrete examples\n"
                "3. Include a practical scenario the reader can relate to\n"
                "4. End with a clear strategic takeaway"
            )

    # ──────────────────────────────────────────
    # 4-Layer structure per H2
    # ──────────────────────────────────────────

    def _build_four_layer(self, intent_type: str, content_type: str, eeat_signal: str, language: str = "vi") -> str:
        """Build the 4-layer structure requirement for each H2 section.

        Labels are translated to the article language so the LLM does not
        output English sub-headings inside non-English articles.
        """
        lb = _get_labels(language)

        # Layer 1: Context framing — varies by intent
        layer_1 = {
            "informational": f"Layer 1 — {lb['L1']}: Open with a specific fact, data point, or industry observation that frames why this topic matters RIGHT NOW. No generic intros.",
            "commercial": f"Layer 1 — {lb['problem']}: Open by naming the exact problem or challenge the reader faces. Be specific — not 'many businesses struggle with X' but 'businesses spending $X/month on Y often see Z problem'.",
            "comparison": f"Layer 1 — {lb['decision']}: Open by establishing what criteria matter for this comparison and why the reader needs to evaluate these options now.",
            "transactional": f"Layer 1 — {lb['value']}: Open by stating the primary benefit or outcome the reader will achieve. Be specific and measurable.",
        }.get(intent_type, f"Layer 1 — {lb['context']}: Open with a specific, relevant observation that hooks the reader.")

        # Layer 2: Core explanation — varies by eeat_signal
        layer_2 = {
            "experience": f"Layer 2 — {lb['L2']} ({lb['experience']}): Explain through first-hand perspective. Describe the actual process, common pitfalls people encounter, and real outcomes. Use 'in practice' language.",
            "expertise": f"Layer 2 — {lb['L2']} ({lb['expertise']}): Provide deep technical/strategic breakdown. Explain the underlying mechanism, process, or framework. Use precise terminology.",
            "authority": f"Layer 2 — {lb['L2']} ({lb['authority']}): Present findings with specific data, statistics, or structured comparisons. Every claim must be backed by evidence.",
            "trust": f"Layer 2 — {lb['L2']} ({lb['trust']}): Present both benefits AND risks/limitations transparently. Include counter-arguments and address common concerns honestly.",
        }.get(eeat_signal, f"Layer 2 — {lb['L2']}: Deep, technical or strategic explanation of the topic.")

        # Layer 3: Practical scenario — always required
        layer_3_base = f"Layer 3 — {lb['L3']} (MANDATORY): "
        if content_type in ("comparison_table", "data_table"):
            layer_3 = layer_3_base + "Include a comparison table with specific criteria and ratings. Follow the table with a brief analysis of what the data shows."
        elif content_type in ("case_study", "product_solution"):
            layer_3 = layer_3_base + "Present a concrete case: Problem → Action taken → Specific result (with numbers). Not hypothetical — must feel real and specific."
        elif content_type == "step_by_step":
            layer_3 = layer_3_base + "Walk through a real example applying each step. Show what happens at each stage with specific details."
        else:
            layer_3 = layer_3_base + "Describe a specific scenario: '[Type of user] doing [specific action] achieves [measurable result] because [reason]'. Must include at least one concrete number or metric."

        # Layer 4: Strategic takeaway — varies by intent
        layer_4 = {
            "informational": f"Layer 4 — {lb['L4']}: What should the reader DO with this knowledge? Give one clear, actionable recommendation with expected outcome.",
            "commercial": f"Layer 4 — {lb['decision_guidance']}: Based on the analysis, what should different reader segments do? Provide segmented advice: 'If you're A, then X. If B, then Y.'",
            "comparison": f"Layer 4 — {lb['recommendation']}: Provide a nuanced recommendation. Not 'X is best' but 'X is optimal for [scenario] because [specific reason], while Y serves [different scenario] better.'",
            "transactional": f"Layer 4 — {lb['next_step']}: What's the immediate next action? Make it concrete and low-friction. Remove decision anxiety.",
        }.get(intent_type, f"Layer 4 — {lb['takeaway']}: One clear, actionable insight the reader can apply immediately.")

        return f"""{layer_1}

{layer_2}

{layer_3}

{layer_4}"""

    # ──────────────────────────────────────────
    # Tone selection for rhythm
    # ──────────────────────────────────────────

    def _select_tone(
        self, section_index: int, total_sections: int,
        content_type: str, eeat_signal: str,
    ) -> dict:
        """Select a tone profile that creates rhythm across sections.

        Avoids all sections having the same analytical tone.
        """
        # Content type overrides
        if content_type in ("comparison_table", "data_table"):
            return _TONE_PROFILES[4]  # data_driven
        if content_type in ("case_study",):
            return _TONE_PROFILES[1]  # narrative
        if content_type in ("technical_breakdown",):
            return _TONE_PROFILES[2]  # technical

        # EEAT signal suggestions
        eeat_tone_map = {
            "experience": 1,   # narrative
            "expertise": 2,    # technical
            "authority": 4,    # data_driven
            "trust": 3,        # contrarian (balanced/challenging)
        }

        if eeat_signal in eeat_tone_map:
            preferred_idx = eeat_tone_map[eeat_signal]
            # But vary: alternate between preferred and cycling
            if section_index % 3 == 0:
                return _TONE_PROFILES[preferred_idx]

        # Cycling for variety — ensures no 2 consecutive sections have the same tone
        tone_idx = section_index % len(_TONE_PROFILES)
        return _TONE_PROFILES[tone_idx]

    # ──────────────────────────────────────────
    # Mandatory elements
    # ──────────────────────────────────────────

    def _get_mandatory_elements(
        self, intent_type: str, content_type: str, eeat_signal: str,
    ) -> list[str]:
        """Determine what elements MUST be present in this section."""
        elements = [
            "At least 1 specific number, metric, percentage, or timeframe",
            "At least 1 practical example or concrete scenario",
        ]

        # Intent-specific
        if intent_type == "commercial":
            elements.append("Problem-solution-bridge structure")
            elements.append("At least 1 comparison (feature, metric, or approach)")
        elif intent_type == "comparison":
            elements.append("Structured comparison (table or clear A vs B)")
            elements.append("Counter-argument or acknowledged trade-off")
        elif intent_type == "transactional":
            elements.append("Specific benefit with measurable outcome")
            elements.append("Objection addressed proactively")

        # EEAT-specific
        if eeat_signal == "trust":
            elements.append("Risk or limitation honestly discussed")
            elements.append("Counter-argument or alternative viewpoint")
        elif eeat_signal == "authority":
            elements.append("Data-backed claim with specific source type (study, report, case data)")
        elif eeat_signal == "experience":
            elements.append("First-hand perspective or real scenario with outcome")

        # Content-type specific
        if content_type in ("comparison_table", "data_table"):
            elements.append("Markdown comparison table with 3+ rows")
        elif content_type == "step_by_step":
            elements.append("Numbered steps with specific actions")

        return elements

    # ──────────────────────────────────────────
    # Anti-patterns
    # ──────────────────────────────────────────

    def _get_anti_patterns(self, section_index: int, total_sections: int) -> list[str]:
        """Generate position-specific anti-patterns to avoid repetitive writing."""
        base_anti = [
            "Do NOT start with a definition or 'X là gì' pattern",
            "Do NOT use 'Việc ... giúp ...' pattern to start paragraphs",
            "Do NOT use generic authority phrases: 'theo chuyên gia', 'nhiều nghiên cứu cho thấy' without specifics",
            "Do NOT end with 'Vì vậy, ...' or 'Do đó, ...' patterns for every paragraph",
            "Do NOT use filler phrases: 'Hiện nay', 'Trong thời đại số', 'Không thể phủ nhận'",
        ]

        # Position-specific
        if section_index == 0:
            base_anti.append("First section: Do NOT open with background/history — jump straight into value")
        if section_index == total_sections - 1:
            base_anti.append("Last body section: Do NOT summarize — leave that for the conclusion")
        if section_index > 0:
            base_anti.append(
                "Do NOT repeat statistics or examples from previous sections — "
                "each section must have UNIQUE data points"
            )

        # Vary opening patterns
        opening_bans = [
            "Do NOT start this section with a question",
            "Do NOT start this section with a statistic",
            "Do NOT start this section with a scenario/story",
            "Do NOT start this section with a claim/statement",
        ]
        # Ban one pattern per section (cycling)
        base_anti.append(opening_bans[section_index % len(opening_bans)])

        return base_anti

    def format_for_prompt(self, strategy: dict) -> str:
        """Format the complete strategy into a prompt block for SectionWriter.

        Args:
            strategy: Output from get_section_strategy()

        Returns:
            Formatted string ready to inject into the section prompt
        """
        parts = []

        parts.append(f"═══ DEPTH STRATEGY ═══\n{strategy['depth_instruction']}")

        parts.append(
            f"\n═══ 4-LAYER STRUCTURE (MANDATORY — these are WRITING GUIDELINES, NOT headings to output) ═══\n"
            f"{strategy['four_layer_structure']}\n\n"
            f"CRITICAL: Write these 4 layers as natural flowing content. "
            f"Do NOT create sub-headings or titles for each layer. "
            f"The layers define your writing FLOW, not the article STRUCTURE."
        )

        tone = strategy["tone_profile"]
        parts.append(f"\n═══ WRITING TONE: {tone['name'].upper()} ═══\n{tone['instruction']}")

        mandatory = strategy["mandatory_elements"]
        if mandatory:
            items = "\n".join(f"  ✓ {e}" for e in mandatory)
            parts.append(f"\n═══ MANDATORY ELEMENTS (must include ALL) ═══\n{items}")

        anti = strategy["anti_patterns"]
        if anti:
            items = "\n".join(f"  ✗ {a}" for a in anti)
            parts.append(f"\n═══ ANTI-PATTERNS (NEVER do these) ═══\n{items}")

        return "\n".join(parts)
