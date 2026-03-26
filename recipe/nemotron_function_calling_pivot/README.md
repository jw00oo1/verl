# Nemotron RL Agentic Function Calling Pivot v1 (verl recipe)

`nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1`를 **verl 학습 스키마로 먼저 정규화한 뒤 parquet로 저장**하는 구조입니다.

> 일부 환경에서는 `datasets.load_dataset("json", ...)`에서 혼합 타입 에러가 날 수 있어, JSONL은 파이썬으로 직접 파싱합니다.

## 포함 파일

- `prepare_dataset.py`: 권장 메인 파이프라인 (JSONL -> verl train/val parquet)
- `convert_jsonl_to_parquet.py`: 단일 parquet 변환 (`--format verl` 기본, `--format raw` 옵션)
- `analyze_dataset.py`: 원본 구조 분석 + parquet round-trip 체크
- `reward_function.py`: GLM4.7 유사 tool-call 포맷용 커스텀 reward
- `rl_dataset.py`: row별 tools를 chat template에 주입하는 커스텀 RLHFDataset
- `train_grpo_vllm.sh`: vLLM 기반 GRPO 예시


> tool chat template 적용을 위해 `train_grpo_vllm.sh`는 `NemotronToolAwareRLHFDataset`을 사용합니다.

## 권장 사용 (학습용)

```bash
python3 recipe/nemotron_function_calling_pivot/prepare_dataset.py \
  --output_dir ~/data/nemotron_fc_pivot
```

이 경로가 가장 실용적입니다. (학습 스키마로 바로 변환 후 저장)

## 단일 parquet 변환

### verl 포맷(기본)

```bash
python3 recipe/nemotron_function_calling_pivot/convert_jsonl_to_parquet.py \
  --format verl \
  --verify_reload \
  --output_parquet ~/data/nemotron_fc_pivot/train.parquet
```

### raw 보관 포맷(옵션)

```bash
python3 recipe/nemotron_function_calling_pivot/convert_jsonl_to_parquet.py \
  --format raw \
  --verify_reload \
  --output_parquet ~/data/nemotron_fc_pivot/raw_train.parquet
```

## 분석

```bash
python3 recipe/nemotron_function_calling_pivot/analyze_dataset.py --check_to_parquet
```

## Reward 규칙

- 정답이 `message` 타입이면: 모델이 tool call을 생성하지 않으면 reward=1 (본문 비교 안 함)
- 정답이 `function_call` 타입이면: 모든 tool call의 함수명/개수/인자 일치 시 reward=1

Tool call 파싱 포맷:

```text
<tool_call>{function-name}
<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>
<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>
...</tool_call>
```
