"""Flatten a table's HTML into a retrieval-friendly text paragraph."""

from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.domain.prompts.flatten_table import FLATTEN_TABLE_PROMPT


def flatten_table(litellm_client: LitellmClient, html_content: str) -> str:
    """
    Convert an HTML table into a dense text paragraph suitable for embedding.

    Used by the retrieval layer to create searchable text representations
    of tables that preserve all technical values.
    """
    resp = litellm_client.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical assistant that transforms technical tables "
                    "into compact but complete summaries. Your goal is to produce a "
                    "single paragraph that retains all essential information from the "
                    "table, so nothing is lost during this flattening process. "
                    "Your outputs will be embedded into a vector index for retrieval."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": FLATTEN_TABLE_PROMPT.format(table_html=html_content),
                    },
                    {
                        "type": "text",
                        "text": f"<html><body>{html_content}</body></html>",
                    },
                ],
            },
        ],
        response_format=None,
        call_type="flatten_table",
    )
    return resp.choices[0].message.content
