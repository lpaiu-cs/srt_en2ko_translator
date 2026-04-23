from __future__ import annotations

import json
import unittest

from build_shipping_failure_corpus import collect_rows


def _row(
    *,
    row_id: str,
    repair_invoked: bool = False,
    repair_accepted: bool = False,
    smaller_block_fallback: bool = False,
    post_wrap_failure: bool = False,
    failure_reasons: dict[str, int] | None = None,
    pre_wrap_failures: dict[str, int] | None = None,
    post_wrap_failures: dict[str, int] | None = None,
    translated_cues: list[dict] | None = None,
) -> dict:
    return {
        "id": row_id,
        "translation_output": {
            "translated_cues": translated_cues or [],
        },
        "pipeline_signals": {
            "repair_invoked": repair_invoked,
            "repair_accepted": repair_accepted,
            "smaller_block_fallback": smaller_block_fallback,
            "post_wrap_failure": post_wrap_failure,
            "failure_reasons": failure_reasons or {},
            "pre_wrap_failures": pre_wrap_failures or {},
            "post_wrap_failures": post_wrap_failures or {},
        },
    }


class ShippingFailureCorpusTests(unittest.TestCase):
    def test_collect_rows_selects_shipping_failures(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "cs231n_sp25_eval_hard40_translated_round37_full_pipeline.jsonl"
            rows = [
                _row(
                    row_id="hard::1",
                    repair_invoked=True,
                    repair_accepted=False,
                    failure_reasons={"english_residual": 1},
                    pre_wrap_failures={"english_residual": 1},
                ),
                _row(
                    row_id="hard::2",
                    smaller_block_fallback=True,
                    post_wrap_failure=True,
                    failure_reasons={"line_overflow": 1},
                    post_wrap_failures={"line_overflow": 1},
                ),
                _row(row_id="hard::3"),
            ]
            input_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

            selected = collect_rows([input_path])
            self.assertEqual([row["id"] for row in selected], ["hard::1", "hard::2"])

            meta1 = selected[0]["shipping_failure_meta"]
            self.assertEqual(meta1["benchmarks"], ["hard40"])
            self.assertEqual(meta1["lanes"], ["full-pipeline"])
            self.assertEqual(meta1["selection_reasons"], ["repair_rejected"])
            self.assertEqual(meta1["failure_families"], ["english_residual"])

            meta2 = selected[1]["shipping_failure_meta"]
            self.assertEqual(meta2["selection_reasons"], ["smaller_block_fallback", "post_wrap_failure"])
            self.assertEqual(meta2["failure_families"], ["fallback_trigger", "wrap_readability"])

    def test_collect_rows_final_postwrap_only_filters_transient_rows(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "cs231n_sp25_eval_hard40_translated_round44_full_pipeline.jsonl"
            rows = [
                _row(
                    row_id="hard::transient",
                    repair_invoked=True,
                    repair_accepted=True,
                    post_wrap_failure=True,
                    failure_reasons={"cps_overflow_severe": 1},
                    post_wrap_failures={"cps_overflow_severe": 1},
                    translated_cues=[
                        {
                            "cue_index": 1,
                            "start": "00:00:00,000",
                            "end": "00:00:02,000",
                            "text": "짧은 문장입니다.",
                        }
                    ],
                ),
                _row(
                    row_id="hard::final",
                    repair_invoked=True,
                    repair_accepted=False,
                    post_wrap_failure=True,
                    failure_reasons={"line_overflow": 1},
                    post_wrap_failures={"line_overflow": 1},
                    translated_cues=[
                        {
                            "cue_index": 2,
                            "start": "00:00:00,000",
                            "end": "00:00:06,000",
                            "text": "첫 줄이 깁니다.\n둘째 줄도 깁니다.\n셋째 줄도 깁니다.",
                        }
                    ],
                ),
            ]
            input_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

            selected = collect_rows([input_path], final_postwrap_only=True)
            self.assertEqual([row["id"] for row in selected], ["hard::final"])
            self.assertEqual(
                selected[0]["shipping_failure_meta"]["final_post_wrap_reasons"],
                ["line_overflow"],
            )


if __name__ == "__main__":
    unittest.main()
