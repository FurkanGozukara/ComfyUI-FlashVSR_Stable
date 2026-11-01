"""Official FlashVSR+ inference engine (OpenImagingLab/FlashVSR_Plus, commit b548a2cc2c).

Vendored for the SECourses FlashVSR CLI. Only the top-level imports were moved under this package
(inference, models, model_io, flashvsr_attn) and the Triton operator probes use the vendored name;
the command-line entry points are not included. Weights: Hugging Face ruchangyao/FlashVSR_plus.
"""

UPSTREAM_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
