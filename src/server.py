import argparse
import os
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MLE-Bench purple A2A server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    skill = AgentSkill(
        id="mle_submission",
        name="Universal MLE-Bench solver",
        description=(
            "Accepts competition.tar.gz plus instructions, uses an OpenAI-powered ML agent "
            "to inspect the dataset, generate a submission, and returns submission.csv."
        ),
        tags=["benchmark", "mle-bench", "kaggle", "llm", "tabular"],
        examples=[],
    )

    agent_card = AgentCard(
        name=os.getenv("AGENT_NAME", "MLE-Bench Universal Purple"),
        description=os.getenv(
            "AGENT_DESCRIPTION",
            "Universal OpenAI-driven ML agent for MLE-Bench competitions",
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        max_content_length=None,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
