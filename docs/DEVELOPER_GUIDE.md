# 개발자 가이드 (Developer Guide)

이 문서는 번역기의 내부 파이프라인 동작 방식, 환경 변수 상세 설정, 평가(Evaluation) 방법 등 개발자 및 기여자를 위한 세부 정보를 제공합니다.

## ⚙️ 번역 파이프라인 (Pipeline)
번역기는 단순히 API를 호출하는 것을 넘어, 자막 품질을 보장하기 위한 다단계 파이프라인을 거칩니다.

1. **동적 블록 생성**: 구두점, 자막 간격(gap), 지속 시간, 소스 길이 등을 기준으로 2~4개의 큐(Cue)를 하나의 번역 블록으로 묶습니다.
2. **Phase 1 (초기 번역)**: 블록을 번역 모델에 전송하고, 큐 개수가 정확히 유지되도록 `emitted_cues` 및 `risk_flags`를 포함한 구조화된 출력(Structured output)을 요구합니다.
3. **구조 검증 및 폴백(Fallback)**: 출력된 스키마와 큐 구조를 검증합니다. 실패 시 Phase 1을 재시도하며, 그래도 실패하면 블록을 더 작게 분할합니다.
4. **Pre-wrap 검증**: 줄 바꿈을 적용하기 전, 자막 정렬, 용어집 적용 여부, 번역되지 않은 영문 잔류물, 앞/뒤 집중도 등을 검사합니다.
5. **줄 바꿈 및 로컬 재조정**: 줄 바꿈을 적용합니다. 만약 Post-wrap 검사에서 실패하면, 우선 자체적인 재줄바꿈(re-wrap) 전략을 시도합니다.
6. **Post-wrap 검증**: 줄 초과(Line overflow), 초당 글자 수(CPS), 어색한 줄 바꿈 위치 등을 최종 검사합니다.
7. **Phase 2 (제한적 복구)**: 위 검증을 통과하지 못한 블록만 Phase 2 복구 모델(기본 `gpt-4o`)에 실패 원인(프롬프트)과 함께 전송하여 수정을 시도합니다.
8. **재귀적 분할**: 복구 후에도 실패하면, 블록을 더 작은 단위로 쪼개어 처음부터 다시 시도합니다.
9. **안전 장치**: 무한 루프를 방지하기 위해 재시도, 복구, 재귀적 분할의 최대 횟수 제한(Depth)이 설정되어 있습니다.

## 🗂 저장소 구조 (Repository Layout)
- `srt_en2ko_translator.py`: 단일 파일 번역을 위한 CLI 엔트리포인트
- `batch_translate_srt.py`: 폴더 기반 배치 번역 실행기
- `subtitle_translator/`: 핵심 패키지 모듈
  - `config.py`: `.env` 로딩 및 런타임 환경 설정
  - `blocks.py`: 동적 블록 생성 및 문맥(Context window) 선택
  - `srt_io.py`: SRT 파일 파싱 및 쓰기
  - `grouping.py`: 문맥 창에 사용되는 문장 그룹화 휴리스틱
  - `glossary.py`: 용어집(Glossary) 저장 및 검색
  - `quality.py`: 구조 검증 및 Pre-wrap/Post-wrap 게이트
  - `translators.py`: 모델 어댑터 및 구조화된 출력 처리
  - `pipeline.py`: 재시도, 복구, 분할 폴백 및 최종 오케스트레이션
- `translation_artifacts/`: 로컬 용어집 로그 및 생성된 아티팩트 저장소

## 🛠 환경 변수 (Environment Variables)
`.env` 파일을 통해 번역기의 다양한 동작을 미세 조정할 수 있습니다.

### 핵심 설정
- `OPENAI_API_KEY`: (필수) OpenAI API 키.
- `SRT_PHASE1_MODEL`: 기본 Phase 1 번역 모델 (기본값: `gpt-4.1-mini`).
- `SRT_REPAIR_MODEL`: 기본 Phase 2 복구 모델 (기본값: `gpt-4o`).
- `SRT_PHASE1_TEMPERATURE`, `SRT_REPAIR_TEMPERATURE`: 생성 온도 설정. 평가(Eval A/B) 실행 시 노이즈를 줄이려면 `0.0`으로 설정하세요.

### 프롬프트 및 컨텍스트
- `SRT_PHASE1_PROMPT_PROFILE`: Phase 1 프롬프트/예제 프로필 (기본값: `fragment_preserving_v2`). 
- `SRT_TRANSLATION_CONTEXT`: 도메인/문맥 힌트. (예: `These subtitles are the Stanford CS231n lecture...`)
- `SRT_TRANSLATION_STYLE`: 어조/스타일 힌트.
- `SRT_USE_CONTEXT_WINDOW`: 소스 문장의 좌/우 문맥을 모델에 제공할지 여부.

### 복구 및 폴백 정책
- `SRT_ENABLE_REPAIR`: 제한된 Phase 2 복구를 활성화합니다.
- `SRT_REPAIR_POLICY`: 복구 정책 변형 (`baseline`, `compact_technical_fragment_v1` 등).
- `SRT_ENGLISH_RESIDUAL_POLICY`: 영문 잔류물 처리 정책 (`coarse`, `technical_split`).
- `SRT_PHASE1_MAX_RETRIES`, `SRT_PHASE2_MAX_REPAIRS`, `SRT_MAX_SPLIT_DEPTH`: 재시도 및 재귀 제어 가드레일.

### 로깅 및 용어집
- `SRT_GLOSSARY_LOG_PATH`: 용어집 JSONL 로그 파일 경로.
- `SRT_METRICS_LOG_PATH`: 파일별 JSONL 지표 로그 파일 경로.
- `SRT_GLOSSARY_MAX_TERMS`: 요청당 주입되는 최대 용어집 항목 수.
- `SRT_ALLOWED_ENGLISH_TERMS`: 결과물에 남아있어도 허용되는 영문 기술 용어 (쉼표로 구분).

### 네트워크 및 타임아웃
- `SRT_REQUEST_TIMEOUT`: 요청 타임아웃(초).
- `SRT_REQUEST_MAX_ATTEMPTS`, `SRT_REQUEST_BACKOFF_MIN_SECONDS`, `SRT_REQUEST_BACKOFF_MAX_SECONDS`: API 속도 제한 및 일시적 오류에 대한 동기식 디버그/평가 재시도 백오프 설정.

### 렌더링 및 가독성 임계값
- `SRT_BLOCK_MIN_CUES`, `SRT_BLOCK_MAX_CUES`, `SRT_BLOCK_MAX_DURATION_MS`, `SRT_BLOCK_MAX_SOURCE_CHARS`, `SRT_BLOCK_MAX_GAP_MS`: 블록 생성기 제어.
- `SRT_MAX_CHARS_PER_LINE`: 라인당 최대 글자 수 (현재 기본값: `28`).
- `SRT_MAX_LINES_PER_CUE`: 큐당 최대 라인 수.
- `SRT_MAX_CPS`: 초당 최대 글자 수 (가독성 기준).
- `SRT_WRAP_POLICY`: 줄 바꿈/가독성 정책 변형 (`baseline`, `cps_relaxed_v1` 등).

## 🧪 평가 (Evaluation)

CS231n 강의 데이터를 사용해 리뷰 세트를 구축하고 평가하는 방법입니다.

### 리뷰 세트 구축
```bash
python3 build_eval_set.py --input-dir cs231n_sp25/eng --output evaluation/cs231n_sp25_eval.jsonl --target-count 40
```

### 파이프라인 리플레이 (동적 블록)
리뷰된 세트에 대해 현재 파이프라인을 다시 실행합니다.
```bash
python3 run_review_eval.py --input evaluation/cs231n_sp25_eval_review_round1.jsonl --output evaluation/cs231n_sp25_eval_translated.jsonl
```

### 프롬프트 A/B 테스트 (고정 블록)
원본 블록 경계를 고정하고 Phase 1 온도를 낮추어 프롬프트 변경 사항만 평가합니다.
```bash
python3 run_review_eval.py --input evaluation/cs231n_sp25_eval_review_round1.jsonl --output evaluation/cs231n_sp25_eval_translated_frozen.jsonl --frozen-blocks --phase1-temperature 0.0 --prompt-profile fragment_preserving_v2
```

### 대규모 Batch 평가 프로세스
API 속도 제한을 피하기 위해 Batch API를 활용하는 평가 방법입니다.

1. Phase 1 준비:
```bash
python3 run_review_eval_batch.py prepare-phase1 \
  --input evaluation/cs231n_sp25_eval_hard40_boundary_aware.jsonl \
  --requests-out evaluation/batch/hard40_phase1_requests.jsonl \
  --manifest-out evaluation/batch/hard40_phase1_manifest.jsonl \
  --phase1-temperature 0.0 \
  --prompt-profile fragment_preserving_v2
```

2. 생성된 requests JSONL을 OpenAI Batch API로 제출/완료 후 결과 다운로드.

3. 엄격한 재시도(Strict-retry) 배치 준비:
```bash
python3 run_review_eval_batch.py prepare-style-retry \
  --phase1-manifest evaluation/batch/hard40_phase1_manifest.jsonl \
  --phase1-output evaluation/batch/hard40_phase1_output.jsonl \
  --requests-out evaluation/batch/hard40_strict_requests.jsonl \
  --manifest-out evaluation/batch/hard40_retry_manifest.jsonl
```

4. 재시도 배치를 다시 OpenAI Batch로 돌린 후, 최종 결과 결합:
```bash
python3 run_review_eval_batch.py finalize \
  --retry-manifest evaluation/batch/hard40_retry_manifest.jsonl \
  --strict-output evaluation/batch/hard40_strict_output.jsonl \
  --output evaluation/cs231n_sp25_eval_hard40_translated_batch.jsonl
```
