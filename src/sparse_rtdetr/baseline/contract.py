"""Contracts shared by the frozen RT-DETRv2 VisDrone baseline adapter."""

from __future__ import annotations

from dataclasses import dataclass

from ..data_protocol.schema import ProtocolContractError


class BaselineContractError(ProtocolContractError):
    """Raised when a baseline adapter contract is violated."""


BASELINE_ID = "rtdetrv2_r18_visdrone_baseline_v1"
NUM_CLASSES = 10
COCO_CATEGORY_IDS = tuple(range(1, 11))
MODEL_LABEL_IDS = tuple(range(10))
ENGINEERING_INPUT_SIZE = (640, 640)
NUM_QUERIES = 300
NUM_TOP_QUERIES = 300

UPSTREAM_R18_DEFAULT_NUM_CLASSES = 80
UPSTREAM_R18_DEFAULT_PARAMETER_COUNT = 20184464
VISDRONE_BASELINE_NUM_CLASSES = 10
VISDRONE_BASELINE_PARAMETER_COUNT = 20094584
CLASS_DEPENDENT_PARAMETER_DELTA = 89880
PARAMETER_COUNT_SCOPE = "random_initialized_no_checkpoint"
PARAMETER_COUNT_RECONCILIATION_STATUS = "UPSTREAM_80_CLASS_VS_VISDRONE_10_CLASS_SEPARATED"

VENDOR_CONFIG_RELATIVE = "vendor/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"
VENDOR_CONFIG_SHA256 = "93182eb778c58346e31b947dec4cb7e673373d3d93210d271a10cc1c6043a1e9"
VENDOR_INCLUDE_RELATIVE = "vendor/rtdetrv2_pytorch/configs/rtdetrv2/include/rtdetrv2_r50vd.yml"
VENDOR_INCLUDE_SHA256 = "57b7fe0920b8c3046f66981f503045023ec68cff0148c88caf7b1e40c9b117c4"
UPSTREAM_COMMIT = "1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47"

R3_ARTIFACT_RELATIVE = "artifacts/data/visdrone_protocol_v2_conversion_r3"
R3_COMPLETION_SHA256 = "46734010937168ac65bf27c3a6f4bf3f234554d5d4a99407903dec4b3ae9e817"
R3_ARTIFACT_INVENTORY_SHA256 = "aaee01ce4e00749b9db8ac5e6e49cf35875a8b266e9c3efc237bdf0b8b109ae7"
R3_ENTRY_CANONICAL_INVENTORY_SHA256 = "ddf32745302d6095e1c3dde89b1a85f05db53e6b4cc8ee85638e0d45bfa8d982"
R3_CONFIG_SHA256 = "58be8659d6d7ae6faace82b36d45523f80cd48bae1bbcc6da4586c53056ec0d0"
R3_CATEGORY_CONTRACT_SHA256 = "b4b309f357cbe130a505a610dff340cc498dc74f766384c4559b2acd900728a0"
R3_SOURCE_IDENTITY_SHA256 = "f0f16ba4438b51a09a6203f78884b199f8e2a2d3fb46309c357368a47a552bb9"

R3_PROCESS_CANONICAL_INVENTORY_SHA256 = "5b1bc8e6888760c55342757a094c2d13c6b5be7a6069df457e44d1094d2a4366"


@dataclass(frozen=True)
class BaselineContract:
    """Typed view of the fields that define this baseline."""

    baseline_id: str = BASELINE_ID
    upstream_implementation: str = "vendored RT-DETRv2 PyTorch"
    backbone: str = "PResNet-18"
    encoder: str = "HybridEncoder"
    decoder: str = "RTDETRTransformerv2"
    decoder_layers: int = 3
    num_queries: int = NUM_QUERIES
    num_classes: int = NUM_CLASSES
    feature_strides: tuple[int, int, int] = (8, 16, 32)
    hidden_dim: int = 256
    num_feature_levels: int = 3
    deformable_sampling_points: tuple[int, int, int] = (4, 4, 4)
    engineering_input_size: tuple[int, int] = ENGINEERING_INPUT_SIZE
    pretrained: bool = False
    checkpoint: None = None
    postprocess: str = "sigmoid_plus_global_top_k"
    num_top_queries: int = NUM_TOP_QUERIES
    nms: bool = False
    train_role: str = "train_core"
    development_role: str = "development"
    confirmatory_role: str = "sealed_and_forbidden"
    test_role: str = "forbidden"
    primary_evaluator: str = "visdrone_official_style_v1"
    secondary_evaluator: str = "coco_secondary_vendor_v1"
    formal_training_configuration_frozen: bool = False
    model_selection_certified: bool = False
    speed_measurement_ready: bool = False
