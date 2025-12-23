# ComfyUI-FlashVSR_Stable
Running FlashVSR on lower VRAM without any artifacts.

## Changelog

#### 23.12.2025
- **🔍 NEW: Pre-Flight Resource Calculator**: Intelligent VRAM/RAM estimation before processing.
  - Automatically calculates estimated VRAM requirements based on resolution, frames, and settings.
  - Provides optimal settings recommendations if OOM is predicted.
  - Example: "Current chunk_size (16) is too high for 8GB VRAM. Recommended: chunk_size=4, resize_factor=0.5"
- **🔧 Unified Processing Pipeline**: `tiny-long` optimizations now applied to all modes (`full`, `tiny`, `tiny-long`).
- **📊 5 VAE Options**: Expanded VAE selection with full support for:
  - `Wan2.1`: Original baseline VAE
  - `Wan2.2`: Updated normalization for Wan2.2 training regime
  - `LightVAE_W2.1`: ~50% VRAM reduction with LightX2V architecture
  - `TAE_W2.2`: Temporal autoencoder for Wan2.2
  - `LightTAE_HY1.5`: HunyuanVideo compatible temporal autoencoder
- **⬇️ Auto-Download**: VAE models automatically download from HuggingFace if missing.
- **🛡️ 95% VRAM Threshold**: OOM recovery only triggers at 95% VRAM usage (optimized for RTX 5070 Ti).
- **✅ Fixed Tensor Permutation**: Resolved video corruption issues with correct tensor shape handling.
- **✅ Fixed Black Borders**: Cropping now happens ONLY AFTER VAE decode is complete.

#### 22.12.2025
- **🚀 NEW: Wan2.2 VAE Support**: Integrated Wan2.2 VAE with optimized normalization statistics for improved video quality.
- **⚡ NEW: LightX2V VAE Integration**: Added LightX2V VAE option for ~50% VRAM reduction and 2-3x faster inference.
- **New Feature**: Added `vae_model` selection in both Init Pipeline and Ultra-Fast nodes.
- **VRAM Optimization**: LightX2V VAE reduces peak VRAM usage by approximately 50% while maintaining near-original quality.
- **Performance**: LightX2V provides 2-3x faster VAE decode times compared to full Wan VAE.
- **Architecture**: All new VAE types maintain full backward compatibility with existing Wan2.1 weights.
- **Documentation**: Updated README with VAE type comparison and usage guidelines.

#### 08.12.2025
- **Optimization**: Ported all VRAM optimizations and Tiled VAE support to `tiny-long` mode, ensuring feature parity across all modes.
- **Performance**: Optimized for Windows environments (speed improvements).
- **Update**: General codebase cleanup and synchronization.

#### 07.12.2025
- **VRAM Optimization**: Implemented auto-fallback for `process_chunk`. If OOM occurs, it automatically retries with `tiled_vae=True` and then `tiled_dit=True`, preventing crashes.
- **Critical Fix**: Fixed a bug in the non-tiled processing path where output was undefined.
- **Optimization**: Defer VAE loading in `full` mode until strictly necessary, significantly reducing peak VRAM usage.
- **Optimization**: Added a proactive 90% VRAM usage warning.
- **Refactor**: Rewrote progress bar to use single-line in-place updates (`\r`) for cleaner console output.
- **Defaults**: Updated default settings for `FlashVSR Ultra-Fast` node to be safer for 16GB cards (`unload_dit=True`, `tiled` options enabled).
- **Bug Fix**: Fixed `AttributeError` in `full` mode by adding a fallback mechanism to manually load the VAE model if the model manager fails.
- **Bug Fix**: Fixed the progress bar to correctly display status in ComfyUI using the `cqdm` wrapper. Added text-based progress bar to logs.
- **Sync**: Enabled VAE spatial tiling for `tiny` mode, bringing VRAM savings from `tiny-long` to the standard fast pipeline.
- **Documentation**: Expanded tooltips for all node parameters and added detailed usage instructions to README.
- **New Feature**: Added `frame_chunk_size` option to split large videos into chunks, enabling processing of large files on limited VRAM by offloading to CPU.
- **Enhancement**: Improved logging to show detailed resource usage (RAM, Peak VRAM, per-step timing) and model configuration details.
- **Optimization**: Added `torch.cuda.ipc_collect()` for better memory cleanup.
- **New Feature**: Added `attention_mode` selection with support for `flash_attention_2`, `sdpa`, `sparse_sage`, and `block_sparse` backends.
- **Refactor**: Cleaned up code and improved error handling for imports.

#### 06.12.2025
- **Bug Fix**: Fixed a shape mismatch error for small input frames by implementing correct padding logic.
- **Optimization**: VRAM is now immediately freed at the start of processing to prevent OOM errors.
- **New Feature**: Added `enable_debug` option for extensive logging.
- **New Feature**: Added `keep_models_on_cpu` option to keep models in RAM (CPU) instead of VRAM.
- **Enhancement**: Added accurate FPS calculation and peak VRAM reporting.
- **Optimization**: Replaced `einops` operations with native PyTorch ops.
- **Optimization**: Added "Conv3d memory workaround".

#### 24.10.2025
- Added long video pipeline that significantly reduces VRAM usage when upscaling long videos.

#### 22.10.2025
- Replaced `Block-Sparse-Attention` with `Sparse_Sage`.
- Added support for running on RTX 50 series GPUs.

#### 21.10.2025
- Initial release of this project.

## Preview
![](./workflow/image1.png)

## Sample Workflow
[Download Workflow JSON](./workflow/FlashVSR.json)

## Performance & VRAM Optimization

This node is optimized for various hardware configurations. Here are some guidelines:

### VRAM Tiers & Settings

| VRAM | Mode | Tiling | Chunk Size | Precision | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **24GB+** | `full` or `tiny` | Disabled | 0 (All) | `bf16`/`auto` | Max quality/speed. |
| **16GB** | `tiny` | `tiled_vae=True` | 0 or ~100 | `bf16`/`auto` | Enable `keep_models_on_cpu`. |
| **12GB** | `tiny` | `tiled_vae=True`, `tiled_dit=True` | ~50 | `fp16` | Use `sparse_sage` attention. |
| **8GB** | `tiny-long` | **Required** | ~20 | `fp16` | Must use tiling and chunking. |

### Performance Enhancements
- **Attention Mode**: Use `sparse_sage_attention` for the best balance of speed and memory. `flash_attention_2` is faster but requires specific hardware/installation.
- **Precision**: `bf16` (BFloat16) is recommended for RTX 3000/4000/5000 series. It is faster and preserves dynamic range better than `fp16`.
- **Chunking**: Use `frame_chunk_size` to process videos in segments. This moves processed frames to CPU RAM, preventing VRAM saturation on long clips.
- **Resize Input**: If the input video is large (e.g., 1080p), use the `resize_factor` parameter to reduce input size to `0.5x` before processing. This drastically reduces VRAM usage and allows for 4x upscaling of the resized result (net 2x output). For small videos, leave at `1.0`.

### Pre-Flight Resource Check (NEW)

Before processing, FlashVSR now performs an intelligent pre-flight check that:

1. **Estimates VRAM Requirements**: Calculates approximate VRAM needed based on resolution, frames, scale, and settings.
2. **Checks Available Resources**: Uses `torch.cuda.mem_get_info()` for accurate real-time VRAM availability.
3. **Provides Recommendations**: If OOM is predicted, suggests optimal settings.

Example console output:
```
============================================================
🔍 PRE-FLIGHT RESOURCE CHECK
💻 RAM: 15.4GB / 95.8GB
💾 VRAM Available: 14.2GB
📊 Estimated VRAM Required: 12.8GB
✅ Safe to proceed. Estimated ~12.8GB needed, 14.2GB available.
============================================================
```

If VRAM is insufficient:
```
⚠️ Current settings require ~18.5GB but only 8.0GB available.
💡 Recommended Optimal Settings:
  • chunk_size = 32
  • tiled_vae = True
  • tiled_dit = True
  • resize_factor = 0.6
```

### VAE Type Comparison

| VAE Type | VRAM Usage | Speed | Quality | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Wan2.1** | 8-12 GB | Baseline | ⭐⭐⭐⭐⭐ | Maximum quality, 24GB+ VRAM |
| **Wan2.2** | 8-12 GB | Baseline | ⭐⭐⭐⭐⭐ | Improved normalization for Wan2.2 models |
| **LightVAE_W2.1** | 4-5 GB | 2-3x faster | ⭐⭐⭐⭐ | 8-16GB VRAM, speed priority |
| **TAE_W2.2** | 6-8 GB | 1.5x faster | ⭐⭐⭐⭐ | Temporal consistency priority |
| **LightTAE_HY1.5** | 3-4 GB | 3x faster | ⭐⭐⭐⭐ | HunyuanVideo compatible, minimum VRAM |

**Recommendations:**
- **8GB VRAM**: Use `LightVAE_W2.1` or `LightTAE_HY1.5` + `tiled_vae=True` + `tiled_dit=True`
- **12GB VRAM**: Use `LightVAE_W2.1` or `Wan2.1` with tiling enabled
- **16GB+ VRAM**: Use any VAE type; `Wan2.1`/`Wan2.2` for maximum quality
- **Speed Priority**: Use `LightVAE_W2.1` or `LightTAE_HY1.5` for 2-3x faster inference
- **Auto-Download**: All VAE files auto-download from HuggingFace if not present

## Node Features

Hover over any input in ComfyUI to see these details:

- **model**: Select the FlashVSR model version.
- **mode**:
  - `tiny`: Standard fast mode. Now supports VAE tiling.
  - `tiny-long`: Streaming mode for very long videos. Lowest VRAM spike.
  - `full`: Uses the full VAE encoder (optional). Highest VRAM. Supports VAE tiling.
- **vae_model**: Select the VAE architecture (5 options with auto-download):
  - `Wan2.1`: Original VAE (default, maximum compatibility)
  - `Wan2.2`: Updated normalization for Wan2.2 training regime
  - `LightVAE_W2.1`: Optimized lightweight VAE (~50% less VRAM, 2-3x faster)
  - `TAE_W2.2`: Temporal autoencoder for better temporal consistency
  - `LightTAE_HY1.5`: HunyuanVideo compatible, minimum VRAM
- **scale**: Upscaling factor (2x or 4x).
- **color_fix**: Corrects color shifts using wavelet transfer. Highly recommended.
- **tiled_vae**: Spatially splits frames during decoding. Saves massive VRAM at the cost of speed.
- **tiled_dit**: Spatially splits frames during diffusion. Crucial for large resolutions (e.g. 4k output).
- **tile_size / overlap**: Controls tile granularity. Smaller tiles = less VRAM but slower.
- **unload_dit**: Aggressively unloads the DiT model before VAE decode.
- **frame_chunk_size**: Splits the temporal dimension. Process N frames at a time.
- **enable_debug**: Prints detailed per-step logs, VRAM stats, and timing to the console.
- **keep_models_on_cpu**: Offloads models to system RAM when idle.
- **attention_mode**: Selects the underlying attention kernel.

## Installation

#### nodes: 

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/naxci1/ComfyUI-FlashVSR_Stable.git
python -m pip install -r ComfyUI-FlashVSR_Stable/requirements.txt
```
📢: For Turing or older GPUs, please install `triton<3.3.0`:  

```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

#### models:

- Download the entire `FlashVSR` folder with all the files inside it from [here](https://huggingface.co/JunhaoZhuang/FlashVSR) and put it in the `ComfyUI/models` directory.

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## Acknowledgments
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) @Wan-Video
- [LightX2V](https://github.com/ModelTC/LightX2V) @ModelTC
- [LightX2V Autoencoders](https://huggingface.co/lightx2v/Autoencoders) @lightx2v
