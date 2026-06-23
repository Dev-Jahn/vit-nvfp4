# SP2 (간소): NVFP4 PTQ 프론트엔드 — DINOv2 수직 슬라이스

- **날짜**: 2026-06-23 · **상태**: 간소 spec→구현 (SP1 패턴 확립됨)
- **선행**: SP1 완료 — `nvfp4_linear`(W4A4 GEMM, torch `_scaled_mm_v2` 백엔드) 사용 가능.
- **목표**: HF `nn.Linear` 트리를 정책 기반으로 `QuantLinear`(W4A4 NVFP4)로 스왑하고, DINOv2에서 양자화 모델 출력이 BF16 대비 정확함을 수치로 검증.

## 스코프
**In:** `QuantLinear`(가중치 정적 양자화, activation 온라인 양자화 → `nvfp4_linear`), 정책 기반 모델 스왑, BF16 대비 정확도 진단(레이어/최종 출력 cosine), DINOv2-base 수직 슬라이스.
**Out (후속):** 다른 패밀리(SigLIP2/Qwen3-VL/V-JEPA: SP2 확장), calibration(SP3), attention 커널(SP-A), 경량 복원(SP5).

## 정밀도 정책 (DINOv2)
- **W4A4 양자화**: 중간 블록의 `attention.attention.{query,key,value}`, `attention.output.dense`, `mlp.{fc1,fc2}`.
- **BF16 유지**: 첫/끝 **2블록**, LayerNorm(norm1/norm2), Dinov2LayerScale, patch-embed Conv, 임베딩, 최종 layernorm. attention 내적/softmax/AV는 SDPA(비-Linear)라 자동 BF16.
- DINOv2-base: 12층 → 양자화 대상 = 층 2..9 (8블록 × 6 Linear = 48개).

## 컴포넌트 (`src/vit_nvfp4/ptq/`)
- `qlinear.py` — `QuantLinear(nn.Module)`: buffers `w_codes`(uint8 N,K), `w_block_scale`(e4m3 N,K//16), `w_global_scale`(fp32), `bias`. `from_linear(nn.Linear)` 정적 양자화. `forward(x)`: (...,K)→flatten→`nvfp4_linear`→reshape. K%16=0 가정(assert).
- `policy.py` — `vit_block_policy(num_layers, skip_first=2, skip_last=2)` → `(name,module)->bool`. 이름에서 `encoder.layer.{i}.` 인덱스 파싱, 중간 블록의 nn.Linear만 True.
- `convert.py` — `quantize_model(model, should_quantize)`: named_modules 순회, 대상 nn.Linear을 부모에서 `QuantLinear.from_linear`로 치환. 치환 수 반환.
- `diagnostics.py` — `output_cosine(ref_model, quant_model, inputs)` 최종 hidden cosine; `per_layer_cosine`(선택, forward hook).

## 성공 기준
1. `QuantLinear`: 랜덤 `nn.Linear` 대비 출력이 `nvfp4_linear` 참조와 일치, BF16 대비 cosine > 0.99(FP4 오차 수준).
2. `quantize_model`: DINOv2-base에서 정확히 48개 Linear 치환, 나머지(첫/끝 2블록·norm·head) 유지.
3. **통합**: 양자화 DINOv2-base의 `last_hidden_state`가 BF16 모델 대비 **cosine ≥ 0.99**(랜덤 pixel_values, sm_120). 이게 W4A4 PTQ 파이프라인의 첫 정확도 신호.

## 구현 순서 (TDD)
1. `QuantLinear` + 테스트(랜덤 Linear 정합성).
2. `policy` + `convert` + 테스트(DINOv2 치환 수/대상 검증, CPU에서 구조만).
3. `diagnostics` + 통합 테스트(DINOv2 quant vs bf16 cosine, sm_120).
4. 진단 스크립트 `examples/dinov2_ptq.py`로 실측 cosine 리포트.

## 메모
- activation 온라인 양자화는 호스트(per-16 amax + to_blocked) — SP1에서 확인된 경로. calibration(정적 activation global scale)은 SP3; SP2는 동적 global(입력별)로 시작.
- 정확도 미달 시 promote 대상(첫/끝 블록 수↑, 민감 레이어 BF16)은 SP4 진단 드라이버에서 체계화.
