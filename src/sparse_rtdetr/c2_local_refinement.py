"""Fixed-Q local box residuals from the same backbone forward's C2.

B0 pools C2 once; B1 reads native C2. Neither changes class logits, DN,
auxiliary predictions, query selection, or the criterion.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from sparse_rtdetr.baseline.training_v2b_deterministic_sampling import bilinear_grid_sample


class C2LocalBoxRefiner(nn.Module):
    def __init__(self, mode: str, *, seed: int = 20261001,
                 projection_channels: int = 32, hidden_channels: int = 64):
        super().__init__()
        if mode not in {"B0", "B1"}:
            raise ValueError(f"unsupported C2 arm: {mode}")
        self.mode = mode
        self.enabled = True
        # Constructors consume only an isolated CPU RNG stream. In particular,
        # attaching the module must not change the resumed augmentation stream.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.projection = nn.Linear(64, projection_channels)
            self.hidden = nn.Linear(9 * projection_channels + 256 + 4, hidden_channels)
            self.residual = nn.Linear(hidden_channels, 4)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
        axis = torch.tensor([-1.0 / 3.0, 0.0, 1.0 / 3.0])
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("offsets", torch.stack((x, y), -1).reshape(9, 2), persistent=False)
        self._c2 = None
        self._query = None

    def image_points(self, boxes: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
        reference = boxes.detach().float()
        points = reference[..., None, :2] + reference[..., None, 2:] * self.offsets.float()
        h, w = image_hw
        return points * points.new_tensor([w, h])

    def sampling_grid(self, boxes: torch.Tensor, image_hw: tuple[int, int],
                      feature_hw: tuple[int, int]) -> torch.Tensor:
        points = self.image_points(boxes, image_hw)
        # PResNet-d's symmetric padded convolutions/maxpool place C2 centers at
        # image edge-coordinate 0.5 + 4*j. Averaging adjacent centers adds 2 px.
        stride, origin = (8.0, 2.5) if self.mode == "B0" else (4.0, 0.5)
        indices = (points - origin) / stride
        fh, fw = feature_hw
        return 2.0 * (indices + 0.5) / points.new_tensor([fw, fh]) - 1.0

    def sample(self, c2: torch.Tensor, boxes: torch.Tensor,
               image_hw: tuple[int, int]) -> torch.Tensor:
        feature = F.avg_pool2d(c2, kernel_size=2, stride=2) if self.mode == "B0" else c2
        grid = self.sampling_grid(boxes, image_hw, tuple(feature.shape[-2:]))
        # CUDA grid_sample has no BF16 kernel. Sampling in FP32 also fixes the
        # coordinate precision independently of autocast; its costs are included.
        with torch.autocast(device_type=c2.device.type, enabled=False):
            samples = bilinear_grid_sample(feature.float(), grid.float())
        return samples.permute(0, 2, 3, 1)  # B,Q,9,C2

    def forward(self, c2: torch.Tensor, query: torch.Tensor, boxes: torch.Tensor,
                image_hw: tuple[int, int]) -> torch.Tensor:
        if boxes.shape[-2:] != (300, 4) or query.shape[-2:] != (300, 256):
            raise ValueError("refinement is restricted to final normal Q300")
        samples = self.sample(c2, boxes, image_hw)
        projected = F.relu(self.projection(samples)).flatten(-2)
        reference = boxes.detach()
        fused = torch.cat((projected, query, reference), dim=-1)
        delta = self.residual(F.relu(self.hidden(fused)))
        scale = torch.cat((reference[..., 2:], reference[..., 2:]), dim=-1)
        # No inverse_sigmoid/sigmoid round trip. At initialization this is the
        # identity in both values and gradients of every original parameter.
        return (boxes + 0.1 * delta.tanh() * scale).clamp(0.0, 1.0)

    def capture_c2(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        if self.enabled:
            self._c2 = output

    def capture_query(self, _module: nn.Module, inputs: tuple, _output: torch.Tensor) -> None:
        if self.enabled:
            self._query = inputs[0]

    def refine_output(self, _module: nn.Module, inputs: tuple, output: dict[str, Any]):
        if not self.enabled:
            return output
        c2, query = self._c2, self._query
        self._c2 = self._query = None
        if c2 is None or query is None:
            raise RuntimeError("C2/final-query capture missing; incompatible backbone/decoder")
        split = output.get("dn_meta", {}).get("dn_num_split", [0, 300])
        if split[-1] != 300 or query.shape[1] != sum(split):
            raise RuntimeError("normal/DN query partition differs from the fixed Q300 contract")
        refined = self(c2, query[:, -300:], output["pred_boxes"], tuple(inputs[0].shape[-2:]))
        return {**output, "pred_boxes": refined}


def attach_c2_refiner(model: nn.Module, mode: str, **kwargs) -> C2LocalBoxRefiner:
    """Attach only after original strict model/optimizer/EMA restoration.

    Attach separately to the already-created online and EMA models, never
    deepcopy live captures. Old state_dict names and optimizer IDs stay intact.
    """
    if hasattr(model, "local_box_refiner"):
        raise ValueError("a local refiner is already attached")
    if (model.decoder.num_queries != 300 or len(model.decoder.dec_bbox_head) != 3
            or model.decoder.decoder.eval_idx != 2):
        raise ValueError("expected original A3b Q300 / three decoder layers / final-layer eval")
    refiner = C2LocalBoxRefiner(mode, **kwargs)
    model.add_module("local_box_refiner", refiner)
    model.backbone.res_layers[0].register_forward_hook(refiner.capture_c2)
    model.decoder.dec_bbox_head[-1].register_forward_hook(refiner.capture_query)
    model.register_forward_hook(refiner.refine_output)
    return refiner


def append_refiner_optimizer_group(optimizer: torch.optim.Optimizer,
                                    refiner: C2LocalBoxRefiner) -> dict[str, Any]:
    """Copy the original default decay group settings; create no moment state."""
    base = next((g for g in optimizer.param_groups if g.get("name") == "default"),
                optimizer.param_groups[-1])
    group = {key: value for key, value in base.items() if key not in {"params", "param_names", "name"}}
    group["params"] = list(refiner.parameters())
    group["name"] = "local_box_refiner"
    group["param_names"] = ["local_box_refiner." + name for name, _ in refiner.named_parameters()]
    optimizer.add_param_group(group)
    if any(p in optimizer.state for p in refiner.parameters()):
        raise RuntimeError("new branch unexpectedly has optimizer history")
    return {key: value for key, value in group.items() if key != "params"}
