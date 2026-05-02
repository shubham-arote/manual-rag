"""Pipeline orchestrator — runs all 6 steps in sequence."""

import logging
from typing import Dict, List, Optional

from manual_rag_api.config import PipelineConfig
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.infrastructure.pipeline.steps import (
    run_context_step,
    run_image_step,
    run_improve_table_step,
    run_ocr_step,
    run_table_step,
)

logger = logging.getLogger(__name__)

_ALL_STEPS = {
    "ocr":           ("OCR Extraction",         run_ocr_step),
    "improve_table": ("Improve Table Structure", run_improve_table_step),
    "context":       ("Context Metadata",        run_context_step),
    "table":         ("Table Metadata",          run_table_step),
    "image":         ("Image Metadata",          run_image_step),
}

# Client routing per step:
#   vision_client   — context (3-page reasoning, needs best vision model)
#   metadata_client — improve_table, table, image (single-image, can use cheaper model)
#   text_client     — ocr post-processing and any text-only steps
_VISION_STEPS   = {"context"}
_METADATA_STEPS = {"improve_table", "table", "image"}


class PDFProcessor:
    """
    Orchestrates the 5-step PDF extraction pipeline.

    Three LLM clients are created from config:
      vision_client   — complex multi-image reasoning (context step only)
      metadata_client — single-image tasks (table correction, table/image metadata)
                        defaults to vision_model but can be a cheaper model via
                        METADATA_MODEL env var
      text_client     — text-only tasks (table flattening, answer generation)

    All three share the same PipelineConfig and are cost-tracked independently.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vision_client   = LitellmClient(model_name=config.vision_model)
        self.metadata_client = LitellmClient(model_name=config.metadata_model)
        self.text_client     = LitellmClient(model_name=config.text_model)
        self.results: Dict[str, dict] = {}

    def run(self, steps: Optional[List[str]] = None) -> Dict[str, dict]:
        """
        Run the pipeline.

        Args:
            steps: Ordered list of step names to run. None = all steps.
                   Valid names: 'ocr', 'improve_table', 'context',
                                'enhance', 'table', 'image'
        Returns:
            Dict mapping step name → result dict (status, metrics…)
        """
        logger.info("=" * 60)
        logger.info("PDF RAG Pipeline")
        logger.info("=" * 60)
        logger.info(f"PDF          : {self.config.pdf_path}")
        logger.info(f"Output       : {self.config.output_dir}")
        logger.info(f"Vision model : {self.config.vision_model}")
        logger.info(f"Text model   : {self.config.text_model}")
        if self.config.max_pages:
            logger.info(f"Max pages    : {self.config.max_pages}")
        logger.info("=" * 60)

        steps_to_run = list(_ALL_STEPS.keys()) if steps is None else steps

        invalid = [s for s in steps_to_run if s not in _ALL_STEPS]
        if invalid:
            raise ValueError(
                f"Unknown steps: {invalid}. Valid: {list(_ALL_STEPS.keys())}"
            )

        logger.info(f"Running steps: {', '.join(steps_to_run)}")

        for step_name in steps_to_run:
            title, func = _ALL_STEPS[step_name]
            if step_name in _VISION_STEPS:
                client = self.vision_client
            elif step_name in _METADATA_STEPS:
                client = self.metadata_client
            else:
                client = self.text_client
            logger.info(f"\n{'=' * 60}\nStarting: {title}\n{'=' * 60}")
            try:
                result = func(self.config, client)
                self.results[step_name] = result
                if result.get("status") == "error":
                    logger.error(f"Step '{step_name}' failed — continuing.")
                else:
                    logger.info(f"Step '{step_name}' completed ✅")
            except Exception as e:
                logger.error(
                    f"Step '{step_name}' raised exception: {e}", exc_info=True
                )
                self.results[step_name] = {"status": "error", "error": str(e)}

        self._print_cost_summary()
        self._print_pipeline_summary()
        return self.results

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _print_cost_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("Cost Summary")
        logger.info("=" * 60)
        total_cost = 0.0
        for label, client in (
            ("vision",   self.vision_client),
            ("metadata", self.metadata_client),
            ("text",     self.text_client),
        ):
            s = client.get_cost_summary()
            total_cost += s["total_cost"]
            logger.info(
                f"[{label:6s}] cost=${s['total_cost']:.6f} | "
                f"tokens={s['total_tokens']:,} | calls={s['call_count']}"
            )
        logger.info(f"{'TOTAL':8s}  cost=${total_cost:.6f}")

    def _print_pipeline_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)
        icons = {
            "success": "✅",
            "skipped": "⏭️",
            "error":   "❌",
            "partial": "⚠️",
        }
        for step, result in self.results.items():
            status = result.get("status", "unknown")
            logger.info(f"{icons.get(status, '?')} {step}: {status}")
        logger.info("=" * 60)

