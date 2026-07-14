import json
import os

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

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

       
        response = self.client.converse(
            modelId=self.model_id,
            messages=body["messages"]
        )

        output = response["output"]["message"]["content"]

        return output[0]["text"]