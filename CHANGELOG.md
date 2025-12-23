# Changelog

All notable changes to ComfyUI-FlashVSR_Stable will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2025-12-23

### 🚀 New Features

#### Pre-Flight Resource Calculator (FIX 9)
- **Intelligent VRAM Estimation**: Calculates approximate VRAM requirements before processing begins.
- **Hardware Detection**: Uses `torch.cuda.mem_get_info()` for accurate real-time VRAM/RAM availability.
- **Settings Recommender**: If OOM is predicted, automatically suggests optimal settings:
  - Recommended `chunk_size` based on available memory
  - Recommended `resize_factor` for constrained systems  
  - Automatic tiling enablement suggestions
- **Console Output**: Clear pre-flight status with actionable recommendations.

#### 5 VAE Model Options (FIX 1 & 7)
- **Unified `vae_model` Dropdown**: Single input replacing redundant `vae_type` and `alt_vae` fields.
- **Five VAE Options**:

| Selection | File | Architecture | Use Case |
|-----------|------|--------------|----------|
| Wan2.1 | Wan2.1_VAE.pth | WanVideoVAE | Original baseline, maximum compatibility |
| Wan2.2 | Wan2.2_VAE.pth | Wan22VideoVAE | Updated normalization for Wan2.2 regime |
| LightVAE_W2.1 | lightvaew2_1.pth | LightX2VVAE | ~50% VRAM reduction, 2-3x faster |
| TAE_W2.2 | taew2_2.safetensors | TAEW22VAE | Temporal autoencoder for Wan2.2 |
| LightTAE_HY1.5 | lighttaehy1_5.pth | LightTAEHY15VAE | HunyuanVideo compatible |

#### Auto-Download System (FIX 6)
- **Automatic Model Downloads**: VAE files auto-download from HuggingFace if missing.
- **Primary Source**: Uses `hf_hub_download()` from HuggingFace Hub.
- **Fallback**: Falls back to `torch.hub.download_url_to_file()` if HF Hub fails.
- **Repository**: All models sourced from `huggingface.co/lightx2v/Autoencoders`.

#### Unified Processing Pipeline (FIX 10)
- **Shared Optimizations**: All modes (`full`, `tiny`, `tiny-long`) now use the same optimized pipeline.
- **Consistent OOM Handling**: Robust memory management applied globally.
- **Mode-Adaptive**: Pipeline adapts behavior based on selected mode while sharing core logic.

### 🐛 Bug Fixes

#### Video Corruption Fix (FIX 7)
- **Root Cause**: Incorrect tensor permutation during VAE decode caused glitchy/broken output.
- **Solution**: Rewrote `tensor2video()` to properly handle all tensor formats:
  - (B, C, F, H, W) - Batch, Channels, Frames, Height, Width
  - (C, F, H, W) - Channels, Frames, Height, Width
  - (F, C, H, W) - Frames, Channels, Height, Width
- **Validation**: Added `torch.clamp(0.0, 1.0)` to ensure valid output range.

#### Black Border Fix (FIX 3)
- **Root Cause**: Padding applied for VAE alignment (divisible by 8/16) was not being cropped correctly.
- **Strict Algorithm**:
  1. Store `original_sH`, `original_sW` before padding.
  2. Apply `reflect` padding (with `replicate` fallback for small images).
  3. Process through VAE decode.
  4. **CRITICAL**: Crop output back to original dimensions ONLY AFTER decode is 100% complete.
- **Result**: No more black borders on output video.

#### Model Loading Logic Bug (FIX 2)
- **Problem**: Users selecting "LightX2V" saw logs loading "Wan2.1_VAE.pth" instead.
- **Solution**: Implemented STRICT file path mapping with explicit class instantiation.
- **Debug Logging**: Added `selected_model` vs `loaded_model_path` logging for verification.

#### OOM False Positive Fix (FIX 5)
- **Problem**: OOM recovery triggered prematurely, leaving VRAM unused.
- **Solution**: Adjusted VRAM threshold to 95% using `torch.cuda.mem_get_info()`.
- **Target**: Optimized for RTX 5070 Ti and similar 16GB cards.

### ⚡ Performance Improvements

#### VRAM Optimization (FIX 5)
- **95% VRAM Threshold**: OOM recovery only triggers at 95% VRAM usage.
- **Aggressive Garbage Collection**: `torch.cuda.empty_cache()` before/after heavy operations.
- **Memory Profiling**: Enhanced `estimate_vram_usage()` considers mode, chunk_size, and tiling.

#### Lossless Resize (FIX 4)
- **Integer Scaling**: Uses `NEAREST` interpolation for clean integer scaling (0.5, 0.25, etc.).
- **Non-Integer Scaling**: Uses `BICUBIC` for fractional factors.
- **Logging**: Reports which interpolation method is used.

### 🛠 Refactoring

#### Code Structure
- **Explicit VAE Instantiation (FIX 7)**: No more state_dict inspection/guessing for model type.
- **VAE_MODEL_MAP**: Centralized configuration for all 5 VAE options with class, file, URL, dimensions.
- **Summary Logging (FIX 8)**: End-of-processing report with total time, peak VRAM, resolutions.

#### Documentation
- **Inline Comments**: Verbose comments marking each FIX location in code.
- **Docstrings**: Comprehensive docstrings for tensor handling functions.

---

## [1.1.0] - 2025-12-22

### 🚀 New Features

- **Wan2.2 VAE Support**: Integrated Wan2.2 VAE with optimized normalization statistics.
- **LightX2V VAE Integration**: Added LightX2V VAE for ~50% VRAM reduction and 2-3x faster inference.
- **VAE Type Selection**: Added `vae_model` dropdown in both Init Pipeline and Ultra-Fast nodes.
- **Factory Function**: Added `create_video_vae()` for programmatic VAE selection.

### ⚡ Performance

- **VRAM Reduction**: LightX2V reduces peak VRAM usage by approximately 50%.
- **Speed Improvement**: LightX2V provides 2-3x faster VAE decode times.

### 🛠 Refactoring

- **Backward Compatibility**: All new VAE types maintain full compatibility with existing Wan2.1 weights.
- **Architecture Constants**: Added `VAE_FULL_DIM`, `VAE_LIGHT_DIM`, `VAE_Z_DIM` for clarity.

---

## [1.0.3] - 2025-12-08

### ⚡ Performance

- **tiny-long Optimization**: Ported all VRAM optimizations and Tiled VAE support to `tiny-long` mode.
- **Windows Optimization**: Speed improvements for Windows environments.
- **Codebase Cleanup**: General synchronization and cleanup.

---

## [1.0.2] - 2025-12-07

### 🚀 New Features

- **Frame Chunking**: Added `frame_chunk_size` option to split large videos into chunks.
- **Attention Mode Selection**: Support for `flash_attention_2`, `sdpa`, `sparse_sage`, and `block_sparse`.
- **Debug Mode**: Added `enable_debug` option for extensive logging.

### 🐛 Bug Fixes

- **OOM Auto-Recovery**: If OOM occurs, automatically retries with `tiled_vae=True`, then `tiled_dit=True`.
- **Non-Tiled Output**: Fixed bug where output was undefined in non-tiled processing path.
- **Progress Bar**: Fixed display in ComfyUI using `cqdm` wrapper.
- **Full Mode VAE**: Added fallback mechanism to manually load VAE if model manager fails.

### ⚡ Performance

- **Deferred VAE Loading**: In `full` mode, VAE loading is deferred until strictly necessary.
- **90% VRAM Warning**: Added proactive warning when VRAM usage approaches 90%.
- **Memory Cleanup**: Added `torch.cuda.ipc_collect()` for better cleanup.

### 🛠 Refactoring

- **Progress Bar**: Rewrote to use single-line in-place updates (`\r`).
- **Default Settings**: Updated defaults for 16GB cards (`unload_dit=True`, tiled options enabled).
- **Documentation**: Expanded tooltips for all node parameters.

---

## [1.0.1] - 2025-12-06

### 🐛 Bug Fixes

- **Shape Mismatch**: Fixed error for small input frames with correct padding logic.
- **VRAM Cleanup**: VRAM immediately freed at start of processing to prevent OOM.

### 🚀 New Features

- **CPU Offload**: Added `keep_models_on_cpu` option to keep models in RAM.
- **FPS Reporting**: Added accurate FPS calculation and peak VRAM reporting.

### ⚡ Performance

- **Native PyTorch**: Replaced `einops` operations with native PyTorch ops where possible.
- **Conv3d Workaround**: Added memory optimization for Conv3d operations.

---

## [1.0.0] - 2025-10-21

### 🚀 Initial Release

- **Video Super Resolution**: ComfyUI node for FlashVSR video upscaling.
- **Three Modes**: `tiny`, `tiny-long`, and `full` processing modes.
- **Tiling Support**: Spatial tiling for VAE and DiT to reduce VRAM.
- **Sparse_Sage Attention**: Replaced Block-Sparse-Attention for RTX 50 series support.
- **Long Video Pipeline**: Streaming mode for very long videos with minimal VRAM spike.
