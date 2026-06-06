# 🎬 영어 SRT -> 한국어 자막 번역기 (SRT Translator)

자막의 원본 타이밍과 인덱스를 완벽하게 보존하면서 자연스러운 한국어로 번역해주는 도구입니다.

## ✨ 주요 기능
- **완벽한 타이밍 보존**: 원본 영어 자막의 싱크(큐 타이밍과 인덱스)를 그대로 유지합니다.
- **문맥을 고려한 자연스러운 번역**: 문장을 단순히 1:1로 번역하지 않고, 자막의 길이와 구두점을 기준으로 적절한 단위(블록)를 나누어 번역합니다.
- **고품질 자막 생성**: 가독성을 위해 초당 글자 수(CPS)와 줄 바꿈을 자동으로 최적화합니다.
- **자동 오류 복구**: 번역이 어색하거나 자막 길이를 초과하는 경우, 해당 부분만 AI가 다시 번역하여 품질을 높입니다.
- **용어집 지원**: 특정 고유명사나 전문 용어를 일관되게 번역할 수 있습니다.
- **배치 처리**: 폴더 내의 여러 SRT 파일을 한 번에 일괄 번역할 수 있습니다.

## 🚀 시작하기

### 1. 설치 및 준비
Python 환경이 필요합니다. (가상 환경 사용을 권장합니다.)

```bash
# 필수 패키지 설치
python3 -m pip install requests
```

### 2. 환경 변수 설정
`.env.example` 파일을 복사하여 `.env` 파일을 만들고 OpenAI API 키를 입력하세요.

```bash
cp .env.example .env
```
`.env` 파일 내용:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 💻 사용 방법

### 단일 파일 번역
하나의 SRT 파일을 번역할 때 사용합니다.
```bash
python srt_en2ko_translator.py input.srt -o output.ko.srt
```

### 폴더 일괄 번역 (배치 처리)
폴더 내의 모든 SRT 파일을 한 번에 번역합니다. 강의 영상이나 시리즈물 번역 시 용어의 일관성을 유지하는 데 유리합니다.
```bash
python batch_translate_srt.py ./eng_subtitles_folder --skip-existing --recursive
```

## 🛠 더 알아보기 (개발자용 문서)
이 번역기는 내부적으로 복잡한 품질 검증(Quality Gates)과 복구 파이프라인(Repair Pipeline)을 통해 동작합니다. 개발 목적의 상세 설정(프롬프트, 모델 변경 등), 파이프라인 구조, 평가(Evaluation) 가이드는 아래 문서를 참고하세요.

- [👉 개발자 가이드 (Developer Guide)](docs/DEVELOPER_GUIDE.md)
