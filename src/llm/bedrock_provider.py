import json
import os
from collections.abc import Iterator

from src.aws.session import get_bedrock_runtime_client
from src.llm.base import LLMProvider


class BedrockProvider(LLMProvider):
    """
    Amazon Bedrock implementation.
    """

    def __init__(self, model: str):

        self.client = get_bedrock_runtime_client()

        self.model_id = model

    def generate(self, prompt: str) -> str:

        body = {"messages": [{"role": "user", "content": [{"text": prompt}]}]}

        response = self.client.converse(
            modelId=self.model_id, messages=body["messages"]
        )

        output = response["output"]["message"]["content"]

        return output[0]["text"]

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt,
                        }
                    ],
                }
            ]
        }

        response = self.client.converse_stream(
            modelId=self.model_id,
            messages=body["messages"],
        )

        for event in response["stream"]:

            if "contentBlockDelta" not in event:
                continue

            delta = event["contentBlockDelta"]["delta"]

            text = delta.get("text")

            if text is not None:
                yield text
