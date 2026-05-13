from __future__ import annotations

from typing import Iterable

from openai import AsyncAzureOpenAI

from app.models import AnalysisResult, AskVideoResponse, ConversationTurn
from app.settings import get_settings


SYSTEM_PROMPT = """\
You are a store analytics assistant that answers questions ONLY using the YOLO detection \
and tracking data provided below. You must follow these rules strictly:

1. ONLY answer from the analytics context provided. Never use general knowledge or speculation.
2. If the data does not contain the answer, say: "This information is not available in the current analysis."
3. Use plain, simple language — avoid technical jargon. Write as if explaining to a store manager.
4. Always cite specific numbers from the analysis (e.g., "12 visitors entered the Promo zone" not "many visitors").
5. Reference zone names, timestamps, and metrics directly from the YOLO tracking output.
6. Keep answers short (2-4 sentences) and actionable.
7. When discussing anomalies or congestion, explain what it means in practical terms \
   (e.g., "3 people waited more than 30 seconds" rather than "loitering detected").

Remember: every number you cite must come from the detection/tracking data below. \
Do not invent or estimate values not present in the analytics context.
""".strip()


class AegisAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        if (
            not self.settings.azure_openai_api_key
            or not self.settings.azure_openai_endpoint
            or not self.settings.azure_openai_deployment_name
        ):
            self.client: AsyncAzureOpenAI | None = None
            return

        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
        )

    def _build_context(self, result: AnalysisResult) -> str:
        lines = [
            "=== YOLO Detection & Tracking Results ===",
            f"Video ID: {result.video_id}",
            f"Engine: {result.engine}",
            f"Generated At: {result.generated_at}",
            "",
            "--- Overall Metrics (from person detection + tracking) ---",
            f"Total tracked visitors: {result.summary_metrics.visitors}",
            f"Estimated buyers (visited checkout): {result.summary_metrics.estimated_buyers}",
            f"Peak simultaneous people detected: {result.summary_metrics.peak_occupancy}",
            f"Average time spent in store: {result.summary_metrics.avg_dwell_minutes} minutes",
            f"Average queue wait time: {result.summary_metrics.avg_queue_minutes} minutes",
            f"Busiest zone: {result.summary_metrics.busiest_zone}",
            f"Anomaly events flagged: {result.summary_metrics.anomaly_count}",
            "",
            "--- Zone-by-Zone Breakdown (from YOLO tracking) ---",
        ]
        for zone in result.zone_insights:
            lines.append(
                f"  {zone.zone_name}: {zone.entries} people entered, "
                f"avg stay {zone.avg_dwell_minutes} min, "
                f"peak {zone.peak_occupancy} people at once, "
                f"congestion {zone.congestion_score:.0%}, "
                f"engagement {zone.engagement_score:.0%}"
            )

        if result.highlights:
            lines.append("")
            lines.append("--- Key Events Detected ---")
            for highlight in result.highlights[:6]:
                lines.append(
                    f"  [{highlight.timestamp}] {highlight.title}: {highlight.description}"
                )

        if result.anomalies:
            lines.append("")
            lines.append("--- Anomalies (unusual patterns from tracking) ---")
            for anomaly in result.anomalies[:8]:
                lines.append(
                    f"  [{anomaly.timestamp}] {anomaly.zone_name} ({anomaly.risk_level} risk): "
                    f"{anomaly.description} (lasted {anomaly.duration_minutes} min)"
                )

        if result.journeys:
            lines.append("")
            lines.append("--- Sample Person Journeys (tracked paths) ---")
            for journey in result.journeys[:5]:
                lines.append(
                    f"  {journey.label}: entered {journey.entered_at}, left {journey.exited_at}, "
                    f"spent {journey.dwell_minutes} min, visited: {', '.join(journey.zones_visited)}"
                )

        lines.append("")
        lines.append("--- Executive Summary ---")
        lines.append(result.executive_summary)

        return "\n".join(lines)

    def _history_to_messages(self, history: Iterable[ConversationTurn]) -> list[dict[str, str]]:
        return [{"role": turn.role, "content": turn.text} for turn in history]

    async def answer(
        self,
        result: AnalysisResult,
        question: str,
        history: list[ConversationTurn] | None = None,
    ) -> AskVideoResponse:
        if not self.client:
            return AskVideoResponse(
                answer="Azure OpenAI configuration is incomplete. Set the endpoint, API key, and deployment name in backend/.env.",
                supporting_points=["Missing Azure OpenAI configuration"],
                suggested_follow_ups=["How do I configure Azure OpenAI for Ask Video?"],
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(self._history_to_messages(history[-10:]))
        messages.append(
            {
                "role": "user",
                "content": f"YOLO Analysis Data:\n{self._build_context(result)}\n\nQuestion: {question}",
            }
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.azure_openai_deployment_name,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=400,
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return AskVideoResponse(
                answer=f"Azure OpenAI request failed: {exc}",
                supporting_points=["LLM request failed"],
                suggested_follow_ups=["Check the Azure deployment name and network access."],
            )

        supporting_points = [card.body for card in result.top_insights[:3]] or [
            highlight.description for highlight in result.highlights[:3]
        ]
        return AskVideoResponse(
            answer=answer or "No answer was returned by the language model.",
            supporting_points=supporting_points[:3],
            suggested_follow_ups=result.recommended_questions[:3],
        )
