# SP1 설계: W4A4 NVFP4 GEMM op (sm_120)

- **날짜**: 2026-06-23
- **상태**: 설계 승인됨 → spec 작성 → (검토 대기)
- **대상 하드웨어**: 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120 / CC 12.0, 96GB each), CUDA 13.1 (nvcc 13.1.115), driver 590.48.01
- **빌드 타깃**: PyTorch **cu130** 휠 + 시스템 nvcc 13.1, `TORCH_CUDA_ARCH_LIST=12.0a`

---

## 1. 배경 (전체 프로젝트 맥락)

전체 프로젝트 `vit-nvfp4`의 목표는 **범용 Vision Transformer용 NVFP4 W4A4 PTQ + calibration 툴킷**이다. 대상 모델군: SigLIP/SigLIP2, Qwen3-VL vision encoder, DINOv2/DINOv3, V-JEPA2/2.1. 이 spec은 그 토대가 되는 **첫 서브프로젝트(SP1) — W4A4 NVFP4 GEMM 커널 op** 만을 다룬다.

### 1.1 핵심 하드웨어 사실 (load-bearing)
- sm_120(소비자/워크스테이션 Blackwell)은 데이터센터 sm_100과 달리 **`tcgen05`/Tensor Memory(TMEM)가 없다.** FP4 연산은 **SM8x식 warp-level `mma.sync`** 로 제공된다.
- NVFP4 전용 atom: `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...e2m1.e2m1...ue4m3` (block=16, UE4M3 스케일). FP4 처리량은 이론상 ~2× FP8.
- 따라서 네이티브 FP4 연산 가속은 실재하며, 그 경로를 타려면 **W4A4**(activation도 FP4)여야 한다. sm_100용 tcgen05 커널은 전부 비호환.
- raw `mma.sync`를 직접 작성하는 것은 **금물**: m16n8k64 FP4 경로의 per-lane scale-factor(SF) fragment 레이아웃이 PTX ISA에 미공개(NVIDIA forum #364020 open). collective-builder(CUTLASS)나 검증된 라이브러리를 재사용한다.

### 1.2 전체 정밀도 정책 (확정)
| 연산 | 정밀도 |
|---|---|
| QKV proj · attn out-proj · MLP fc1/fc2(/fc3) — 중간 블록 | **W4A4 NVFP4** (activation도 FP4) |
| Attention 내적 QKᵀ · softmax · A·V | **NVFP4 시도 → 실패 시 FP8** (SageAttention-3=FP4 / SageAttention-2++=FP8 계열 탐색) |
| LayerNorm · patch-embed conv · head · pooling · 첫/끝 2블록 · residual · RoPE · LayerScale | **BF16** |

### 1.3 서브프로젝트 로드맵
- **SP1 — W4A4 NVFP4 GEMM op** ← *본 spec*. 수치 정확하고 벤치된 Python 호출 가능 dense FP4 GEMM. 독립 완료.
- SP2 — NVFP4 양자화 코어 + PTQ 프론트엔드 (`QuantLinear`, 2단계 스케일, 모델별 config 분기, `nn.Linear` 스왑).
- SP3 — Calibration 하니스 (forward-hook observer, activation global scale, 패밀리별 calib 데이터).
- SP4 — 정확도 진단 + mixed-precision 드라이버 (레이어별 cosine-sim, end-task 지표, 민감 레이어 BF16/FP8 승격).
- SP-A — Attention 커널 (FP4 시도 / FP8 fallback). SP5 — outlier 완화(Had16·RegCache) + calibration-set 기반 **경량 복원**(블록 reconstruction / 짧은 fine-tune). scratch QAT는 범위 밖.

---

## 2. SP1 스코프

**In scope**
- sm_120에서 동작하는 **dense W4A4 NVFP4 GEMM**을 torch custom op로 노출.
- 정합성 검증 스위트 + 성능 벤치마크.
- NVFP4 양자화/역양자화 유틸, E2M1 패킹, E4M3 block scale, FP32 global scale, 128×4 SF swizzle.
- 재사용 우선 사다리 평가 → 통과하는 백엔드를 균일 op 뒤에 래핑. 전부 실패 시 CUTLASS-79 기반 torch extension 직접 빌드.
- 환경/빌드 토대 (uv 프로젝트, cu130 torch, 빌드 플래그).

**Out of scope (후속 SP)**
- `nn.Linear` 스왑, 모델 통합, calibration 루프, 정확도 진단.
- attention 커널(FP4/FP8). residual·norm·head 등 BF16 경로.
- 가중치 체크포인트 export/import 포맷 변환.

---

## 3. 성공 기준 (검증 가능한 목표)

1. **정합성 게이트** — `nvfp4_gemm`/`nvfp4_linear` 출력이:
   - 동일 양자화 값을 fp32에서 곱한 **FP4-emulated 참조**와 cosine-sim ≥ 0.999, 최대 상대오차가 누적(reduction) 오차 수준 이내.
   - 원본 BF16 행렬곱 참조와 cosine-sim ≥ 0.99 (FP4 양자화 자체 오차는 허용; 목적은 "커널이 의도한 FP4 수학을 정확히 수행"하는지 확인이며 #2577식 all-zeros/garbage를 반드시 검출).
   - 다양한 shape(K%16=0 및 패딩 필요 케이스 포함)과 입력 분포(정규/heavy-tail/outlier 주입)에서 통과.
2. **성능 게이트** — ViT 대표 shape에서 BF16·FP8 대비 TFLOP/s를 측정·기록. FP4가 BF16 대비 유의미한 가속(목표 ≥ 2×, 실패 시 원인 분석)을 보이는지 정량화.
3. **산출물** — import 가능한 패키지(`vit_nvfp4.nvfp4`), `pytest` 정합성 스위트, 벤치 스크립트, 선택된 백엔드와 그 근거를 기록한 짧은 `PHASE0_RESULT.md`.

---

## 4. NVFP4 수치 포맷 & 2단계 스케일 (정확한 정의)

ecosystem(modelopt / torchao / compressed-tensors)에서 검증된 **2단계 스케일**을 그대로 따른다. (MXFP4 — block 32, E8M0 power-of-two, global 없음 — 와 혼동 금지.)

- **원소**: E2M1 (1 sign, 2 exp, 1 mantissa). 표현 가능 크기 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, 최대 `6.0`.
- **블록**: 16 원소마다 하나의 스케일. dtype = **FP8 E4M3 (UE4M3, 최대 448)**.
- **텐서**: 추가로 텐서당 하나의 **FP32 global scale**.

### 4.1 인코드 (양자화)
```
amax_tensor = max(|x|)                          # 텐서(또는 양자화 축) 전체
global_scale (저장, FP32) = amax_tensor / (6 * 448)
s_enc = 1 / global_scale = (6 * 448) / amax_tensor

각 블록 b (16 원소):
  amax_block = max(|x_b|)
  block_scale_e4m3 (저장) = to_e4m3( clamp( (amax_block / 6) * s_enc, e4m3_min, 448 ) )
                              # (amax_block/6)에 s_enc를 곱한 뒤 E4M3로 캐스트하는 coupling이 필수.
                              # amax_block/6 을 그대로 저장하면 안 됨.
각 원소:
  decoded_scale = float(block_scale_e4m3) * global_scale     # ≈ amax_block / 6
  q_e2m1 = round_to_nearest_even_e2m1( x / decoded_scale )   # |·| ≤ 6 로 매핑
```

### 4.2 디코드 (HW tensor core가 수행)
```
x_hat = q_e2m1 * float(block_scale_e4m3) * global_scale
```
global scale를 빠뜨리면 동적범위가 조용히 잘린다.

### 4.3 패킹 & 레이아웃 (틀리면 에러 없이 오답)
- E2M1 2개를 1바이트로 패킹: `(q[...,1::2] << 4) | q[...,0::2]`, torch dtype `float4_e2m1fn_x2`.
- block scale 텐서는 **row-major 불가**. HW GEMM은 **128×4 swizzle** 레이아웃 요구 (torchao `to_blocked()` / FlashInfer `SfLayout.layout_128x4`).
- contraction 차원 K는 **16의 배수**여야 하며, scale row는 128로 패딩.

### 4.4 weight vs activation 스케일의 정적/동적 분리
- **weight**: per-block(16) E4M3 + per-tensor FP32, 전부 **정적**(오프라인, 데이터 불필요). — SP1에서는 테스트용으로 즉석 양자화.
- **activation**: per-tensor FP32 global scale은 **정적**(SP3 calibration이 산출), per-16 block scale은 **동적**(추론 시 각 블록 live amax로 계산). SP1의 `nvfp4_linear`는 activation을 온라인 양자화해 op에 넘긴다.

---

## 5. 아키텍처 / 컴포넌트

### 5.1 양자화 유틸 (백엔드 무관, 순수 torch)
- `format.py` — E2M1/E4M3 상수, 코드 테이블, `round_to_nearest_even_e2m1`.
- `quant.py`
  - `quantize_to_nvfp4(x, axis, block=16) -> (packed_e2m1, block_scale_e4m3, global_scale_fp32)`
  - `dequantize_nvfp4(packed, block_scale, global_scale, shape) -> fp32` — **정합성 오라클**.
- `pack.py` — E2M1 pack/unpack, `to_blocked()`(128×4 swizzle)와 역변환, K 패딩.

### 5.2 op 인터페이스 (균일, 백엔드 dispatch)
- 저수준: `nvfp4_gemm(a_packed, a_block_scale, a_global_scale, b_packed, b_block_scale, b_global_scale, *, out_dtype=torch.bfloat16) -> Tensor`
- 상위: `nvfp4_linear(x_bf16, w_packed, w_block_scale, w_global_scale, *, x_global_scale=None, bias=None) -> Tensor`
  - `x_global_scale`가 주어지면 정적, 없으면 입력에서 동적 산출. activation 온라인 양자화 → swizzle → `nvfp4_gemm` 호출 → (옵션) bias 가산.
- `torch.library` custom op으로 등록(autograd 불필요, inference 전용; 단 후속 SP5 경량 복원을 위해 fake-quant STE 경로는 SP2/SP5에서 추가).

### 5.3 백엔드 — 재사용 우선 사다리
동일 정합성 테스트로 각 단을 게이트. 통과하는 **첫 단**을 기본 백엔드로 선택, 나머지는 교차검증용으로 유지. 환경변수/인자로 강제 선택 가능.

1. **FlashInfer** `flashinfer.gemm.mm_fp4(a, b, a_descale, b_descale, out_dtype=bf16, block_size=16, backend='b12x')`. sm_120 cubin **JIT 컴파일** 예상(precompiled 미배포 #3294). 첫 호출 컴파일 지연/캐싱 확인.
2. **CUTLASS 예제 79b** (`79_blackwell_geforce_gemm`, `nv_float4_t<float_e2m1_t>`, `arch::Sm120`, `Sm1xxBlkScaledConfig`) → torch C++/CUDA extension. **반드시** `TORCH_CUDA_ARCH_LIST=12.0a` + #2906 정렬 픽스(`alignas(64)` TMA-descriptor Params, SF smem `cute::array_aligned`/16B 정렬).
3. **cuBLASLt** block-scaled FP4 (`CUDA_R_4F_E2M1` 데이터 + `CUDA_R_UE4M3` VEC16 스케일) 소형 바인딩. sm_120 공식 문서 불명확 → on-device 검증.
4. **vLLM** `cutlass_scaled_fp4_mm`(sm120 커널) — 교차검증/대조.

**전부 실패 시 직접 빌드**: CUTLASS-79 collective builder 기반 kernel + torch custom-op 바인딩을 작성(`cuda` / `cute-dsl-ref` 스킬 활용). 동일 op 인터페이스 유지.

### 5.4 정합성 오라클 (두 단계 참조로 버그/양자화오차 분리)
- **FP4-emulated 참조**: 같은 `quantize_to_nvfp4` 출력으로 `dequantize_nvfp4` 후 fp32 matmul → 커널이 의도한 FP4 수학을 정확히 했는지(버그 검출).
- **BF16 참조**: 원본 텐서 bf16 matmul → 양자화 총오차 가늠.

### 5.5 fallback 빌드 경로 (조건부)
- `csrc/` 아래 CUTLASS-79 파생 커널 + pybind/`torch.utils.cpp_extension`. CUTLASS 4.x 체크아웃 핀. 빌드 스크립트는 `TORCH_CUDA_ARCH_LIST=12.0a` 강제.

---

## 6. 환경 / 빌드

- `/home/jahn/workspace/vit-nvfp4`에 **uv** 프로젝트(`pyproject.toml`). 시스템 python·repvis venv 재사용 안 함.
- **PyTorch cu130** 휠(승인됨) — 시스템 CUDA 13.1 / nvcc 13.1과 정합. (repvis의 cu128과 별개 환경.)
- `TORCH_CUDA_ARCH_LIST=12.0a` **필수** — 'a' 접미사 누락 시 NVFP4 block-scaled MMA가 조용히 깨짐(pytorch #172807). (후속 grouped/MoE 불필요 — ViT tower는 dense.)
- 의존성 핀(초기): FlashInfer ≥ 0.6.7, CUTLASS 4.x(서브모듈/체크아웃), (선택) torchao·modelopt는 SP2부터. 빌드 도구: cmake/ninja(존재), nvcc 13.1(존재).

---

## 7. 알려진 리스크 / 함정 (테스트 계획에 반영)

| # | 리스크 | 출처 | 대응 |
|---|---|---|---|
| R1 | FlashInfer cutlass 백엔드가 sm_120에서 all-zeros / cudnn `GraphNotSupported` | issue #2577 (open) | 정합성 게이트로 검출, b12x 우선 |
| R2 | CUTLASS 79a가 CUDA 13.1에서 misaligned-address 크래시 | issue #2906 (커뮤니티 픽스 미머지) | 정렬 픽스 적용 후 빌드 |
| R3 | precompiled sm_120 cubin 미배포 → JIT 의존 | issue #3294 | 첫 호출 컴파일 지연·캐시 측정, nvcc 정합 확인 |
| R4 | arch 'a' 접미사 자동 strip → NVFP4 무력화 | pytorch #172807 | `TORCH_CUDA_ARCH_LIST=12.0a` 명시 |
| R5 | scale swizzle(128×4) 미적용 → 에러 없이 오답 | 연구 | `to_blocked()` 강제, 오라클로 검출 |
| R6 | torch cu130 ↔ 빌드 toolkit 불일치 | 환경 | cu130 + nvcc 13.1 정합 확인(첫 과제) |

---

## 8. 제안 파일 구조 (SP1 범위)

```
vit-nvfp4/
├── pyproject.toml                 # uv, torch cu130
├── docs/superpowers/specs/2026-06-23-nvfp4-w4a4-gemm-design.md
├── src/vit_nvfp4/
│   └── nvfp4/
│       ├── __init__.py
│       ├── format.py              # E2M1/E4M3 상수, rounding
│       ├── quant.py               # quantize/dequantize_to_nvfp4
│       ├── pack.py                # E2M1 packing, 128x4 swizzle, K padding
│       ├── gemm.py                # nvfp4_gemm / nvfp4_linear (백엔드 dispatch)
│       └── backends/
│           ├── __init__.py        # 가용성 probe + 선택
│           ├── reference.py       # fp32 emulated 오라클
│           ├── flashinfer_b12x.py
│           ├── cublaslt.py
│           └── cutlass79.py       # (조건부) extension 래퍼
├── csrc/                          # (조건부) CUTLASS-79 파생 커널 소스
├── tests/test_gemm_correctness.py
├── bench/bench_gemm.py
└── PHASE0_RESULT.md               # 선택된 백엔드 + 정합/성능 결과 기록
```

---

## 9. 테스트 계획

- **단위**: `quantize→dequantize` round-trip 오차가 FP4 격자 한계 이내. pack/unpack 역가역성. swizzle/unswizzle 역가역성.
- **정합성**: §5.4 두 참조 대비. shape 매트릭스 {M ∈ 256,1024,4096; K,N ∈ 768,1024,1152,1408,3072,4304,6144}, K%16≠0 패딩 케이스 포함. 입력 분포 {정규, heavy-tail, outlier 토큰 주입}.
- **회귀**: 선택 백엔드 외 가용 백엔드와 상호 일치(허용오차 내) 교차검증.
- 모든 정합성 테스트는 **실제 RTX PRO 6000 위에서** 실행(에뮬레이션 불가 — R1~R3는 device-specific).

## 10. 벤치 계획

- `bench/bench_gemm.py`: 위 shape 매트릭스에서 NVFP4 vs BF16(`torch.matmul`) vs FP8(`torch._scaled_mm`) TFLOP/s, warmup·다회 측정·중앙값. 결과를 `PHASE0_RESULT.md`에 표로 기록.

## 11. 미해결 질문 (Phase-0가 경험적으로 답함)

1. 이 박스(sm_120 + CUDA 13.1 + driver 590)에서 **어떤 백엔드가 수치 정확한가** — 사다리 1~4 중 통과자. (없으면 직접 빌드.)
2. CUTLASS-79 정렬 픽스가 CUDA 13.1 / CUTLASS 4.x에서 충분한가, sm_120a 추가 버그는 없는가.
3. FlashInfer b12x JIT가 이 nvcc 13.1에서 컴파일되는가, 첫 호출 지연/캐시 거동.
4. activation per-block scale을 GEMM이 내부 동적 계산하는가, 아니면 host에서 swizzle된 채로 미리 넘겨야 하는가 (런타임 경로·지연 영향).
5. cu130 torch 휠 가용성/안정성과 nvcc 13.1 extension 빌드 정합.
