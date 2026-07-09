"""
[3] Topic Cluster Architect
=============================
Creates structured topic clusters: 1 Pillar + 5-10 Cluster articles.

Each article has:
- Clear role in the cluster
- Defined anchor link text
- No intent overlap

Output:
- Pillar page definition
- Cluster article list with roles & relationships
- Internal linking map
"""

import logging

from core.llm.llm_client import llm
from core.llm.prompt_templates import PromptTemplates
from core.models import (
    TopicCluster,
    ClusterKeyword,
    Keyword,
    ClusterRole,
)
from core.database import get_session

logger = logging.getLogger(__name__)


class ClusterArchitect:
    """Topic Cluster Architect - Designs pillar + cluster content structures."""

    def design_cluster(
        self,
        pillar_keyword: str,
        serp_analysis: dict,
        intent_analysis: dict,
        num_clusters: int = 7,
    ) -> dict:
        """
        Design a complete topic cluster around a pillar keyword.

        Args:
            pillar_keyword: The main keyword for the pillar page
            serp_analysis: Output from SerpEngine
            intent_analysis: Output from IntentMapper
            num_clusters: Number of cluster articles (5-10)

        Returns:
            Complete cluster blueprint with pillar + cluster articles
        """
        logger.info(f"[Cluster] Designing cluster for pillar: '{pillar_keyword}'")

        # Step 1: Generate cluster blueprint using LLM
        blueprint = self._generate_blueprint(
            pillar_keyword, serp_analysis, intent_analysis, num_clusters
        )

        # Step 2: Validate no intent overlap
        blueprint = self._validate_no_overlap(blueprint)

        # Step 3: Design internal linking map
        blueprint["linking_map"] = self._design_linking_map(blueprint)

        # Step 4: Save to database
        self._save_to_db(blueprint)

        logger.info(
            f"[Cluster] Created cluster: 1 pillar + "
            f"{len(blueprint.get('clusters', []))} cluster articles"
        )

        return blueprint

    def _generate_blueprint(
        self,
        pillar_keyword: str,
        serp_analysis: dict,
        intent_analysis: dict,
        num_clusters: int,
    ) -> dict:
        """Generate the cluster blueprint using LLM."""

        system_prompt = PromptTemplates.cluster_blueprint_system(num_clusters)

        gaps = intent_analysis.get("intent_gaps", [])
        angles = intent_analysis.get("suggested_angles", [])
        related = serp_analysis.get("related_searches", [])
        paa = serp_analysis.get("people_also_ask", [])

        user_prompt = f"""Pillar Keyword: "{pillar_keyword}"

Current SERP Intent: {serp_analysis.get('dominant_intent', 'informational')}
Saturation Score: {serp_analysis.get('saturation_score', 50)}/100
Avg Word Count in SERP: {serp_analysis.get('avg_word_count', 1500)}

Intent Gaps Found:
{chr(10).join(f'- {g["gap"]} (opportunity: {g.get("opportunity_score", "?")}/10)' for g in gaps)}

Suggested Angles:
{chr(10).join(f'- {a.get("title_suggestion", a.get("angle_description", str(a)))}' for a in angles)}

Related Searches:
{chr(10).join(f'- {r}' for r in related[:10])}

People Also Ask:
{chr(10).join(f'- {q["question"]}' for q in paa[:10])}

Design a topic cluster with 1 pillar page and {num_clusters} cluster articles.
Ensure each cluster article fills a different gap and has no intent overlap."""

        result = llm.chat_json(system_prompt, user_prompt, temperature=0.5)
        result["pillar_keyword"] = pillar_keyword
        return result

    def _validate_no_overlap(self, blueprint: dict) -> dict:
        """Validate and adjust cluster to prevent intent overlap."""

        clusters = blueprint.get("clusters", [])
        if len(clusters) <= 1:
            return blueprint

        cluster_summary = "\n".join(
            f"{i}. [{c.get('content_type', '?')}] {c.get('title', 'Untitled')} "
            f"(Intent: {c.get('target_intent', '?')}) — {c.get('intent_differentiation', 'N/A')}"
            for i, c in enumerate(clusters)
        )

        user_prompt = f"""Pillar: {blueprint.get('pillar', {}).get('title', 'Unknown')}

Cluster Articles:
{cluster_summary}

Check for intent overlap between any articles and suggest adjustments if needed."""

        validation = llm.chat_json(
            PromptTemplates.CLUSTER_OVERLAP_CHECK_SYSTEM, user_prompt, temperature=0.3
        )

        if validation.get("has_overlap") and validation.get("adjustments"):
            for adj in validation["adjustments"]:
                idx = adj.get("index", 0)
                if 0 <= idx < len(clusters):
                    clusters[idx]["title"] = adj.get("new_title", clusters[idx]["title"])
                    clusters[idx]["intent_differentiation"] = adj.get(
                        "new_differentiation", clusters[idx].get("intent_differentiation", "")
                    )
            logger.info(f"[Cluster] Adjusted {len(validation['adjustments'])} overlapping articles")

        blueprint["clusters"] = clusters
        return blueprint

    def _design_linking_map(self, blueprint: dict) -> dict:
        """Design the internal linking structure for the cluster."""
        pillar = blueprint.get("pillar", {})
        clusters = blueprint.get("clusters", [])

        linking_map = {
            "pillar_to_clusters": [],
            "clusters_to_pillar": [],
            "cross_cluster_links": [],
        }

        for i, cluster in enumerate(clusters):
            linking_map["pillar_to_clusters"].append({
                "from": pillar.get("title", "Pillar"),
                "to": cluster.get("title", f"Cluster {i+1}"),
                "anchor_text": cluster.get("pillar_anchor", cluster.get("keyword", "")),
            })

        for i, cluster in enumerate(clusters):
            linking_map["clusters_to_pillar"].append({
                "from": cluster.get("title", f"Cluster {i+1}"),
                "to": pillar.get("title", "Pillar"),
                "anchor_text": cluster.get("anchor_text", pillar.get("keyword", "")),
            })

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1 = clusters[i]
                c2 = clusters[j]
                if c1.get("target_intent") != c2.get("target_intent"):
                    linking_map["cross_cluster_links"].append({
                        "from": c1.get("title", f"Cluster {i+1}"),
                        "to": c2.get("title", f"Cluster {j+1}"),
                        "anchor_text": c2.get("keyword", ""),
                        "bidirectional": True,
                    })

        return linking_map

    def _save_to_db(self, blueprint: dict):
        """Save topic cluster to database."""
        session = get_session()
        try:
            pillar = blueprint.get("pillar", {})
            pillar_kw = blueprint.get("pillar_keyword", pillar.get("keyword", ""))

            tc = TopicCluster(
                name=pillar.get("title", pillar_kw),
                description=pillar.get("description", ""),
                pillar_keyword=pillar_kw,
            )
            session.add(tc)
            session.flush()

            kw = session.query(Keyword).filter_by(keyword=pillar_kw).first()
            if not kw:
                kw = Keyword(keyword=pillar_kw)
                session.add(kw)
                session.flush()

            ck = ClusterKeyword(
                cluster_id=tc.id,
                keyword_id=kw.id,
                role=ClusterRole.PILLAR,
                suggested_title=pillar.get("title", ""),
                anchor_text=pillar_kw,
                intent_differentiation="Main pillar page",
            )
            session.add(ck)

            for cluster in blueprint.get("clusters", []):
                cluster_kw_text = cluster.get("keyword", "")
                ckw = session.query(Keyword).filter_by(keyword=cluster_kw_text).first()
                if not ckw:
                    ckw = Keyword(keyword=cluster_kw_text)
                    session.add(ckw)
                    session.flush()

                ck = ClusterKeyword(
                    cluster_id=tc.id,
                    keyword_id=ckw.id,
                    role=ClusterRole.CLUSTER,
                    suggested_title=cluster.get("title", ""),
                    anchor_text=cluster.get("anchor_text", ""),
                    intent_differentiation=cluster.get("intent_differentiation", ""),
                )
                session.add(ck)

            session.commit()
            logger.info(f"[Cluster] Saved cluster '{tc.name}' with {len(blueprint.get('clusters', []))} articles")
        except Exception as e:
            session.rollback()
            logger.error(f"[Cluster] DB save error: {e}")
        finally:
            session.close()
