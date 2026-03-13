"""Canonical model download specs and required-model policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from state.app_state_types import ModelFileType


@dataclass(frozen=True, slots=True)
class ModelFileDownloadSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool
    repo_id: str
    description: str

    @property
    def name(self) -> str:
        return self.relative_path.name

# #OLD CODE
# MODEL_FILE_ORDER: tuple[ModelFileType, ...] = (
#     "checkpoint",
#     "upsampler",
#     "text_encoder",
#     "zit",
# )
# # END OF OLD CODE

# NEW CODE
MODEL_FILE_ORDER: tuple[ModelFileType, ...] = (
    "checkpoint",
    "upsampler",
    "text_encoder",
    # "zit",
)
# END OF NEW CODE

DEFAULT_MODEL_DOWNLOAD_SPECS: dict[ModelFileType, ModelFileDownloadSpec] = {
    # # OLD CODE
    # "checkpoint": ModelFileDownloadSpec(
    #     relative_path=Path("ltx-2.3-22b-distilled.safetensors"),
    #     expected_size_bytes=43_000_000_000,
    #     is_folder=False,
    #     repo_id="Lightricks/LTX-2.3",
    #     description="Main transformer model",
    # ),
    # # END OF OLD CODE

    # # NEW CODE - LTX-2 distilled (43GB Xet format)
    # "checkpoint": ModelFileDownloadSpec(
    #     relative_path=Path("ltx-2-19b-distilled.safetensors"),
    #     expected_size_bytes=43_000_000_000,  # 43.3 GB - Xet format
    #     is_folder=False,
    #     repo_id="Lightricks/LTX-2",
    #     description="Distilled model - 43GB",
    # ),
    # # END OF NEW CODE
    # NEW CODE - LTX-2 distilled (19GB Xet format)


    # "checkpoint": ModelFileDownloadSpec(
    #         relative_path=Path("ltxv-2b-0.9.8-distilled.safetensors"),
    #         expected_size_bytes=6_000_000_000,  # 19 GB - Xet format
    #         is_folder=False,
    #         repo_id="Lightricks/LTX-Video",
    #         description="Distilled model - 6GB",
    #     ),

    "checkpoint": ModelFileDownloadSpec(
        relative_path=Path("ltx-2-19b-dev-fp4.safetensors"),
        expected_size_bytes=19_000_000_000,  # 19 GB - Xet format
        is_folder=False,
        repo_id="Lightricks/LTX-2",
        description="Distilled model - 43GB",
    ),
    # END OF NEW CODE
    "upsampler": ModelFileDownloadSpec(
        relative_path=Path("ltx-2-spatial-upscaler-x2-1.0.safetensors"),
        expected_size_bytes=1_000_000_000,  # ~996 MB
        is_folder=False,
        repo_id="Lightricks/LTX-2",  # Dùng từ LTX-2 thay vì LTX-2.3
        description="2x Spatial Upscaler - 996MB",
    ),
    # "text_encoder": ModelFileDownloadSpec(
    #     relative_path=Path("gemma-3-12b-it-qat-q4_0-unquantized"),
    #     expected_size_bytes=25_000_000_000,
    #     is_folder=True,
    #     repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
    #     description="Gemma text encoder (bfloat16)",
    # ),
    # "text_encoder": ModelFileDownloadSpec(
    #     relative_path=Path("gemma-3-1b-it"),
    #     expected_size_bytes=3_000_000_000,
    #     is_folder=True,
    #     repo_id="google/gemma-3-1b-it",
    #     description="Gemma text encoder",
    # ),
    "text_encoder": ModelFileDownloadSpec(
        relative_path=Path("gemma-3-4b-it"),
        expected_size_bytes=3_000_000_000,
        is_folder=True,
        repo_id="google/gemma-3-4b-it",
        description="Gemma text encoder",
    ),

    # "zit": ModelFileDownloadSpec(
    #     relative_path=Path("Z-Image-Turbo"),
    #     expected_size_bytes=31_000_000_000,
    #     is_folder=True,
    #     repo_id="Tongyi-MAI/Z-Image-Turbo",
    #     description="Z-Image-Turbo model for text-to-image generation",
    # ),
}


# DEFAULT_REQUIRED_MODEL_TYPES: frozenset[ModelFileType] = frozenset(
#     {"checkpoint", "upsampler"}  # Removed "zit" - using API for image generation
# )

DEFAULT_REQUIRED_MODEL_TYPES: frozenset[ModelFileType] = frozenset(
    {"checkpoint", "text_encoder", "upsampler"}  # Removed "zit" - using API for image generation, Removed "upsampler"
)

def resolve_required_model_types(
    base_required: frozenset[ModelFileType],
    has_api_key: bool,
    use_local_text_encoder: bool = False,
) -> frozenset[ModelFileType]:
    if not base_required:
        return base_required
    if has_api_key and not use_local_text_encoder:
        return base_required
    return cast(frozenset[ModelFileType], base_required | {"text_encoder"})
