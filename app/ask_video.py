from __future__ import annotations

from app.models import AnalysisResult, AskVideoResponse, ConversationTurn
from app.agent import AegisAgent

agent = AegisAgent()


async def answer_question(
    result: AnalysisResult,
    question: str,
    history: list[ConversationTurn] | None = None,
) -> AskVideoResponse:
    return await agent.answer(result, question, history=history)
