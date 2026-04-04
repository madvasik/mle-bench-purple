"""Thin wrapper around the official OpenAI Responses API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


class OpenAIResponsesClient:
    def __init__(self, api_key: str, model: str = "gpt-5.4"):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def create_initial_response(
        self,
        *,
        system_prompt: str,
        user_input: str,
        tools: list[dict[str, Any]],
    ):
        return self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_input,
            tools=tools,
        )

    def create_followup_response(
        self,
        *,
        previous_response_id: str,
        tool_outputs: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ):
        return self.client.responses.create(
            model=self.model,
            previous_response_id=previous_response_id,
            input=tool_outputs,
            tools=tools,
        )

    @staticmethod
    def extract_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    chunks.append(getattr(content, "text", ""))
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    @staticmethod
    def extract_tool_calls(response: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            raw_args = getattr(item, "arguments", "{}") or "{}"
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            tool_calls.append(
                ToolCall(
                    call_id=getattr(item, "call_id"),
                    name=getattr(item, "name"),
                    arguments=parsed_args,
                )
            )
        return tool_calls
