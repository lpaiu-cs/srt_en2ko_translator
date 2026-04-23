from __future__ import annotations

import unittest

from split_shipping_failure_corpus import filter_rows


class SplitShippingFailureCorpusTests(unittest.TestCase):
    def test_filter_rows_by_family(self) -> None:
        rows = [
            {
                "id": "a",
                "shipping_failure_meta": {"failure_families": ["english_residual", "fallback_trigger"]},
            },
            {
                "id": "b",
                "shipping_failure_meta": {"failure_families": ["wrap_readability"]},
            },
        ]
        selected = filter_rows(rows, "english_residual")
        self.assertEqual([row["id"] for row in selected], ["a"])


if __name__ == "__main__":
    unittest.main()
