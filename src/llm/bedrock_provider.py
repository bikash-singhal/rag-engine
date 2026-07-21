from collections.abc import Iterator
from time import perf_counter

from src.aws.session import get_bedrock_runtime_client
from src.llm.base import LLMProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BedrockProvider(LLMProvider):
    """
    Amazon Bedrock implementation.
    """

    def __init__(self, model: str):

        self.client = get_bedrock_runtime_client()

        self.model_id = model

        logger.info(
            "Initialized Bedrock provider with model: %s",
            self.model_id,
        )

    def generate(self, prompt: str) -> str:

        body = {"messages": [{"role": "user", "content": [{"text": prompt}]}]}

        logger.debug("Sending synchronous request to Bedrock.")

        start = perf_counter()

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=body["messages"],
            )
        except Exception:
            logger.exception("Bedrock synchronous request failed.")
            raise

        elapsed = perf_counter() - start

        logger.debug(
            "Bedrock response received in %.3f sec.",
            elapsed,
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

        logger.debug("Starting Bedrock streaming request.")

        start = perf_counter()

        logger.info("Streaming started.")

        try:
            response = self.client.converse_stream(
                modelId=self.model_id,
                messages=body["messages"],
            )

            chunk_count = 0

            for event in response["stream"]:

                if "contentBlockDelta" not in event:
                    continue

                delta = event["contentBlockDelta"]["delta"]

                text = delta.get("text")

                if text:
                    chunk_count += 1
                    yield text

        except Exception:
            logger.exception("Bedrock streaming request failed.")
            raise

        elapsed = perf_counter() - start

        logger.debug(
            "Streaming completed in %.3f sec (%d chunks).",
            elapsed,
            chunk_count,
        )
