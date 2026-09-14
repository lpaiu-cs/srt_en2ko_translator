from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import batch_translate_srt as folder_batch
import run_review_eval as sync_eval
import run_review_eval_batch as batch_eval
from subtitle_translator.config import load_runtime_config
from subtitle_translator.glossary import GlossaryStore
from subtitle_translator.metrics import TranslationMetrics
from subtitle_translator.models import Cue, EmittedCue, PhaseTranslationResult, TranslationBlock, TranslationRequest
from subtitle_translator.openai_batch import OpenAIBatchClient
from subtitle_translator.pipeline import (
    _run_phase1_style_retry, _run_phase1_with_retry, _run_phase2_repair,
    _translate_block_recursive, translate_srt,
)
from subtitle_translator.translators import OpenAIChatTranslator


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.config = load_runtime_config(glossary_log_path="")
        self.config.openai_api_key = "synthetic-test-key"
        self.config.metrics_log_path = None
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        network = patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected network call"))
        network.start()
        self.addCleanup(network.stop)

    def write_rows(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    @staticmethod
    def response(status, payload=None):
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload or {}).encode()
        return response

    def test_transport_errors_escape_all_translation_phases(self):
        block = TranslationBlock([Cue(1, "00:00:00,000", "00:00:03,000", "Hello.")])
        result = PhaseTranslationResult([EmittedCue(1, "안녕하세요.")])
        for error in (requests.HTTPError("401", response=self.response(401)), requests.Timeout("timeout")):
            translator = Mock()
            translator.translate_block.side_effect = error
            translator.repair_block.side_effect = error
            calls = [
                lambda: _run_phase1_with_retry(block, translator, [], self.config),
                lambda: _run_phase2_repair(block, result, ["line_overflow"], translator, [], self.config, None),
                lambda: _run_phase1_style_retry(block, result, [], [], [], [], [], translator, [], None),
            ]
            for call in calls:
                with self.subTest(error=type(error).__name__, phase=calls.index(call)), self.assertRaises(type(error)):
                    call()
            self.assertEqual(translator.translate_block.call_count, 2)
            self.assertEqual(translator.repair_block.call_count, 1)

    def test_multicue_source_fallback_is_counted_and_not_shipped(self):
        cues = [Cue(i + 1, f"00:00:0{i},000", f"00:00:0{i + 1},000", text)
                for i, text in enumerate(["we are looking", "at a simple", "neural network", "for this task"])]
        translator = Mock()
        translator.translate_block.return_value = PhaseTranslationResult([])
        metrics = TranslationMetrics()
        with contextlib.redirect_stderr(io.StringIO()):
            kept = _translate_block_recursive(TranslationBlock(cues), translator, self.config, None, metrics, 0, ())
            self.assertEqual([cue.text for cue in kept], [cue.text for cue in cues])
            self.assertEqual(metrics.source_fallback_cues, 4)
            self.assertEqual(metrics.single_cue_source_fallbacks, 0)
            with self.assertRaisesRegex(RuntimeError, "Translation incomplete"):
                translate_srt(cues, translator, self.config)

    def test_folder_scan_excludes_outputs_and_directories(self):
        (self.root / "nested").mkdir()
        for name in ("a.srt", "a.ko.srt", "UPPER.KO.SRT", "nested/b.srt", "nested/b.ko.srt"):
            (self.root / name).touch()
        (self.root / "directory.srt").mkdir()
        self.assertEqual([p.name for p in folder_batch.find_files(self.root, "*", False)], ["a.srt"])
        self.assertEqual([p.name for p in folder_batch.find_files(self.root, "*", True)], ["a.srt", "b.srt"])

    def run_folder(self, error=None, extra_args=()):
        (self.root / "a.srt").touch()
        (self.root / "b.srt").touch()
        with patch("sys.argv", ["batch", str(self.root), "--retries", "1", *extra_args]), \
             patch.object(folder_batch, "load_runtime_config", return_value=self.config), \
             patch.object(folder_batch, "build_translator", return_value=Mock()), \
             patch.object(folder_batch, "create_glossary_store", return_value=GlossaryStore(None)), \
             patch.object(folder_batch, "process_file", side_effect=error, return_value=(self.root / "out", TranslationMetrics())) as process, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return folder_batch.main(), process.call_count

    def test_folder_exit_status_and_skip_existing(self):
        self.assertEqual(self.run_folder(ValueError("invalid SRT")), (1, 2))
        self.assertEqual(self.run_folder(), (0, 2))
        (self.root / "a.ko.srt").touch()
        self.assertEqual(self.run_folder(extra_args=("--skip-existing",)), (0, 1))

    def test_folder_auth_error_aborts_without_file_retries(self):
        error = requests.HTTPError("unauthorized", response=self.response(401))
        with self.assertRaises(requests.HTTPError):
            self.run_folder(error, ("--retries", "3"))

    def test_incomplete_translation_does_not_overwrite_existing_output(self):
        source = self.root / "a.srt"
        source.write_text("1\n00:00:00,000 --> 00:00:03,000\nHello.\n", encoding="utf-8")
        output = self.root / "a.ko.srt"
        output.write_text("previous good output", encoding="utf-8")
        translator = Mock()
        translator.translate_block.return_value = PhaseTranslationResult([])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(RuntimeError):
            folder_batch.process_file(source, translator, GlossaryStore(None), self.config)
        self.assertEqual(output.read_text(), "previous good output")

    def test_upload_retries_replay_the_same_file_bytes(self):
        payload = b'{"custom_id":"audit-payload"}\n'
        path = self.root / "requests.jsonl"
        path.write_bytes(payload)
        for first_failure in (503, "timeout"):
            with self.subTest(first_failure=first_failure):
                client = OpenAIBatchClient("synthetic", max_attempts=2)
                bodies = []
                def request(method, url, **kwargs):
                    prepared = requests.Request(method, url, files=kwargs["files"], data=kwargs["data"]).prepare()
                    bodies.append(payload in prepared.body)
                    if len(bodies) == 1:
                        if first_failure == "timeout":
                            raise requests.Timeout("simulated")
                        return self.response(503)
                    return self.response(200, {"id": "synthetic-file"})
                with patch.object(client.session, "request", side_effect=request), patch("subtitle_translator.openai_batch.time.sleep"):
                    client.upload_batch_file(path)
                self.assertEqual(bodies, [True, True])

    def test_quota_exhaustion_is_terminal_but_rate_limits_are_retried(self):
        exhausted = self.response(429, {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}})
        translator = OpenAIChatTranslator(self.config)
        request = TranslationRequest(TranslationBlock([Cue(1, "00:00:00,000", "00:00:03,000", "Hello.")]))
        with patch.object(translator.session, "post", return_value=exhausted) as post:
            with self.assertRaises(requests.HTTPError):
                translator.translate_block(request)
            self.assertEqual(post.call_count, 1)
        client = OpenAIBatchClient("synthetic")
        with patch.object(client.session, "request", return_value=exhausted) as get:
            with self.assertRaises(requests.HTTPError):
                client.retrieve_batch("synthetic")
            self.assertEqual(get.call_count, 1)
        with self.assertRaises(requests.HTTPError):
            self.run_folder(requests.HTTPError("quota", response=exhausted))
        valid = {"choices": [{"message": {"content": json.dumps({"emitted_cues": [{"cue_index": 1, "text": "안녕하세요."}], "risk_flags": []})}}]}
        with patch.object(translator.session, "post", side_effect=[self.response(429), self.response(200, valid)]) as post, patch("subtitle_translator.translators.time.sleep"):
            translator.translate_block(request)
            self.assertEqual(post.call_count, 2)

    def entry(self, detector_miss=False):
        texts = (["like model is that it can be trained with just associations", "of images and text."]
                 if detector_miss else ["We would ideally want to be able to use a CLIP model out", "of the box."])
        return {
            "id": "heldout::1", "lecture": "heldout", "source_file": "/nonexistent/original.srt",
            "cue_indices": [240, 241],
            "source_cues": [dict(cue_index=240 + i, start=f"00:00:0{i * 4},000", end=f"00:00:0{(i + 1) * 4},000", text=text)
                            for i, text in enumerate(texts)],
            "previous_source_sentences": ["Frozen previous context."],
            "next_source_sentences": ["Frozen next context."],
            "block_lint": {"lint_actions": ["carry_context_only"] if detector_miss else []},
        }

    def test_frozen_snapshot_preserves_text_timing_order_and_context(self):
        entry = self.entry()
        with patch.object(Path, "read_text", side_effect=AssertionError("must not read live source")):
            block = sync_eval._hydrate_frozen_block(entry)
            batch_block = batch_eval._hydrate_block(entry)
        self.assertEqual(block, batch_block)
        self.assertEqual(block.cues[0].text, entry["source_cues"][0]["text"])
        self.assertEqual(block.cues[0].start, entry["source_cues"][0]["start"])
        self.assertEqual(block.previous_source_sentences, entry["previous_source_sentences"])
        self.assertEqual(block.next_source_sentences, entry["next_source_sentences"])
        entry["cue_indices"].reverse()
        with self.assertRaises(ValueError):
            sync_eval._hydrate_frozen_block(entry)

    def test_sync_frozen_cli_needs_no_original_file_and_runs_final_normalization(self):
        entry = self.entry()
        entry["source_cues"][0]["text"] = "ResNet is useful."
        source = self.write_rows("sync_input.jsonl", [entry])
        output = self.root / "sync_output.jsonl"
        self.config.english_fallback_terms = [{"source": "ResNet", "aliases": ["레스넷"]}]
        def translated(block, *args, **kwargs):
            return [Cue(cue.index, cue.start, cue.end, "레스넷은 유용합니다.") for cue in block.cues]
        with patch("sys.argv", ["eval", "--input", str(source), "--output", str(output), "--frozen-blocks"]), \
             patch.object(sync_eval, "load_runtime_config", return_value=self.config), \
             patch.object(sync_eval, "build_translator", return_value=Mock()), \
             patch.object(sync_eval, "_translate_block_recursive", side_effect=translated), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sync_eval.main(), 0)
        row = json.loads(output.read_text())
        self.assertEqual(row["current_block"]["previous_source_sentences"], entry["previous_source_sentences"])
        self.assertIn("ResNet", row["translation_output"]["translated_cues"][0]["text"])
        self.assertNotIn(self.config.openai_api_key, output.read_text())

    def test_frozen_context_setting_controls_sync_and_batch_request_bodies(self):
        entry = self.entry()
        block = sync_eval._hydrate_frozen_block(entry)
        previous = [EmittedCue(cue.index, "번역된 문장입니다.") for cue in block.cues]
        for enabled in (False, True):
            self.config.use_context_window = enabled
            translator = OpenAIChatTranslator(self.config)
            for mode in ("initial", "full_block", "offending_cue_only"):
                with self.subTest(enabled=enabled, mode=mode):
                    request = TranslationRequest(
                        block=block,
                        strict_style_retry=mode != "initial",
                        strict_retry_mode="full_block" if mode == "initial" else mode,
                        previous_emitted_cues=previous,
                        protected_cue_indices=[240],
                        offending_cue_indices=[241],
                    )
                    response_cues = previous[-1:] if mode == "offending_cue_only" else previous
                    response_key = "offending_cue_rewrites" if mode == "offending_cue_only" else "emitted_cues"
                    content = {response_key: [{"cue_index": cue.cue_index, "text": cue.text} for cue in response_cues], "risk_flags": []}
                    response = self.response(200, {"choices": [{"message": {"content": json.dumps(content)}}]})
                    batch_body = translator.build_phase1_request_body(request)
                    with patch.object(translator.session, "post", return_value=response) as post:
                        translator.translate_block(request)
                    self.assertEqual(post.call_args.kwargs["json"], batch_body)
                    payload = json.loads(batch_body["messages"][-1]["content"])
                    self.assertEqual(payload["left_context"], entry["previous_source_sentences"] if enabled else [])
                    self.assertEqual(payload["right_context"], entry["next_source_sentences"] if enabled else [])
                    self.assertEqual(block.previous_source_sentences, entry["previous_source_sentences"])
                    self.assertEqual(block.next_source_sentences, entry["next_source_sentences"])

    def test_batch_disabled_context_matches_manifest(self):
        self.config.use_context_window = False
        self.check_batch_roundtrip(False)

    def test_batch_request_profile_matches_recorded_profile_and_mode(self):
        for detector_miss in (False, True):
            with self.subTest(detector_miss=detector_miss):
                self.check_batch_roundtrip(detector_miss)

    def check_batch_roundtrip(self, detector_miss):
        context_enabled = self.config.use_context_window
        entry = self.entry(detector_miss)
        source = self.write_rows("source.jsonl", [entry])
        paths = {key: str(self.root / (key + ".jsonl")) for key in ("requests", "manifest", "retry_requests", "retry_manifest", "final")}
        args = batch_eval.build_parser().parse_args(["prepare-phase1", "--input", str(source), "--requests-out", paths["requests"], "--manifest-out", paths["manifest"]])
        with patch.object(batch_eval, "_load_config_from_args", return_value=self.config), contextlib.redirect_stdout(io.StringIO()):
            batch_eval.cmd_prepare_phase1(args)
        manifest_text = Path(paths["manifest"]).read_text()
        self.assertNotIn(self.config.openai_api_key, manifest_text)
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["provenance"]["runtime_config"]["use_context_window"], context_enabled)
        self.assertEqual(manifest["current_block"]["previous_source_sentences"], entry["previous_source_sentences"])
        self.assertEqual(manifest["current_block"]["next_source_sentences"], entry["next_source_sentences"])
        request_body = json.loads(Path(paths["requests"]).read_text())["body"]
        payload = json.loads(request_body["messages"][-1]["content"])
        self.assertEqual(payload["left_context"], entry["previous_source_sentences"] if context_enabled else [])
        self.assertEqual(payload["right_context"], entry["next_source_sentences"] if context_enabled else [])
        translated = (["이 모델의 장점은 단지 연관성만으로도 학습할 수 있다는 점입니다.", "이미지와 텍스트의."]
                      if detector_miss else ["이상적으로는 CLIP 모델을 바로", "바로 사용할 수 있으면 좋겠죠."])
        phase1 = {"emitted_cues": [dict(cue_index=240 + i, text=text) for i, text in enumerate(translated)], "risk_flags": []}
        phase1_output = self.write_rows("phase1_output.jsonl", [{"custom_id": manifest["custom_id"], "response": {
            "status_code": 200, "body": {"choices": [{"message": {"content": json.dumps(phase1)}}]},
        }}])
        args = batch_eval.build_parser().parse_args(["prepare-style-retry", "--phase1-manifest", paths["manifest"], "--phase1-output", str(phase1_output), "--requests-out", paths["retry_requests"], "--manifest-out", paths["retry_manifest"]])
        profiles = []
        original = OpenAIChatTranslator.build_phase1_request_body
        def capture(translator, request):
            profiles.append(translator._effective_prompt_profile(request))
            return original(translator, request)
        with patch.object(batch_eval, "_load_config_from_args", return_value=self.config), patch.object(OpenAIChatTranslator, "build_phase1_request_body", capture), contextlib.redirect_stdout(io.StringIO()):
            batch_eval.cmd_prepare_style_retry(args)
        retry = json.loads(Path(paths["retry_manifest"]).read_text())
        self.assertEqual(profiles, ["fragment_preserving_v3"])
        self.assertEqual(retry["effective_strict_prompt_profile"], profiles[0])
        self.assertEqual(retry["strict_retry_mode"], "offending_cue_only" if detector_miss else "full_block")
        retry_body = json.loads(Path(paths["retry_requests"]).read_text())["body"]
        retry_payload = json.loads(retry_body["messages"][-1]["content"])
        self.assertEqual(retry_payload["left_context"], entry["previous_source_sentences"] if context_enabled else [])
        self.assertEqual(retry_payload["right_context"], entry["next_source_sentences"] if context_enabled else [])
        strict = ({"offending_cue_rewrites": [{"cue_index": 241, "text": "이미지와 텍스트의 연관성으로요."}], "risk_flags": []}
                  if detector_miss else phase1)
        strict_output = self.write_rows("strict_output.jsonl", [{"custom_id": retry["strict_custom_id"], "response": {
            "status_code": 200, "body": {"choices": [{"message": {"content": json.dumps(strict)}}]},
        }}])
        args = batch_eval.build_parser().parse_args(["finalize", "--retry-manifest", paths["retry_manifest"], "--strict-output", str(strict_output), "--output", paths["final"]])
        # Finalization must restore the saved settings, not this changed shell configuration.
        self.config.phase1_prompt_profile = "fragment_preserving_v1"
        self.config.max_chars_per_line = 8
        with patch.object(batch_eval, "_load_config_from_args", return_value=self.config), patch.object(batch_eval, "_candidate_is_overedited", return_value=True), contextlib.redirect_stdout(io.StringIO()):
            batch_eval.cmd_finalize(args)
        final = json.loads(Path(paths["final"]).read_text())
        trace = final["pipeline_signals"]["style_retry_trace"]
        self.assertEqual(trace["effective_strict_prompt_profile"], profiles[0])
        self.assertEqual(trace["strict_candidate_raw_emitted_cues"][0]["text"], translated[0])
        self.assertEqual(final["pipeline_signals"]["style_retry_rejection_causes"], {"overedited_candidate": 1})
        restored = batch_eval._manifest_config(self.config, retry, args)
        self.assertEqual(restored.phase1_prompt_profile, "fragment_preserving_v2")
        self.assertEqual(restored.max_chars_per_line, 28)
        self.config.phase1_prompt_profile = "fragment_preserving_v2"
        self.config.max_chars_per_line = 28


if __name__ == "__main__":
    unittest.main()
