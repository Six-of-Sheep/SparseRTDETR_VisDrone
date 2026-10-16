"""R0 localization-only adaptation of the bound RT-DETRv2/PResNet18 model.

The original attention, feature, selection, classifier and DN implementations
are reused. Only the regression loop/heads change. No LQE or other arm exists.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import importlib
from pathlib import Path

import torch
from torch import nn

from ..localization.distribution import require_finite, validate_reference
from ..localization.consumer_geometry import positive_box


@dataclass(frozen=True)
class R0Config:
    seed: int = 0
    input_hw: tuple[int, int] = (768, 1344)
    physical_batch_size: int = 8
    accumulation_steps: int = 2
    epochs: int = 120
    updates_per_epoch: int = 304
    amp_dtype: str = "bfloat16"
    num_denoising: int = 100
    sampling_backend: str = "deterministic_gather"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_max_norm: float = .1
    warmup_steps: int = 2000
    ema_decay: float = .9999
    ema_warmups: int = 2000

    def validate(self, *, synthetic=False):
        if (self.seed != 0 or self.epochs != 120 or self.updates_per_epoch != 304 or
                self.accumulation_steps != 2 or self.num_denoising != 100 or
                self.sampling_backend != "deterministic_gather" or self.learning_rate != 1e-4 or
                self.weight_decay != 1e-4 or self.clip_max_norm != .1 or self.warmup_steps != 2000 or
                self.ema_decay != .9999 or self.ema_warmups != 2000):
            raise ValueError("R0 recipe differs from the authorized configuration")
        if not synthetic and (self.input_hw != (768, 1344) or self.physical_batch_size != 8 or self.amp_dtype != "bfloat16"):
            raise ValueError("formal R0 must use 768x1344, batch8x2, BF16")
        if synthetic and (min(self.input_hw) < 128 or any(d % 32 for d in self.input_hw) or
                          self.physical_batch_size < 1 or self.amp_dtype not in ("float32", "bfloat16")):
            raise ValueError("invalid synthetic fixture shape")

    def binding(self):
        return {**asdict(self),"input_hw":list(self.input_hw),"arm":"R0","representation":"r0_scalar_sigmoid_go_v1",
                "pre_reference":"first layer pre_box, detached for layer0 scalar regression","reference":"previous decoded box detached for next attention and regression",
                "feature_history":"previous layer features detached, added to current head feature","inverse_sigmoid_eps":1e-5,
                "regression_dim":4,"go_heads":5,"losses":["vfl","bbox","giou"],"lqe":False,"FGL":False,"DDF":False,"endpoint":"e120_EMA",
                "head_initializer":"r0_name_seed_hidden_v1_last_zero","base_initializer":"torch_cpu_seed0_fork_then_verified_ImageNet",
                "training_rng":"after construction reset python/numpy/torchCPU/CUDA to seed0 independently; never resume acceptance",
                "ema_geometry_policy":"immutable_anchor_sampling_R0_constants; average_float_learned_and_BN_state; integer_BN_unchanged",
                "loss_normalization":"per_family_physical_batch_then_equal_mean_of_two","scheduler":"update_warmup2000_cosine_to_1e-6",
                "augmentation_stop_logged_epoch":118,"num_workers":2,"prefetch_factor":2,"grad_scaler":False}


from ..localization.model import row_major_position_embedding


def _vendor():
    return importlib.import_module("src.zoo.rtdetr.rtdetrv2_decoder")


class R0Transformer(nn.Module):
    def __init__(self, base):
        super().__init__()
        for name, value in vars(base).items():
            if not name.startswith("_") and name != "training":
                setattr(self, name, value)
        for name, value in base.named_children():
            self.add_module(name, value)
        for name, value in base.named_buffers(recurse=False):
            self.register_buffer(name, value)
        mlp = _vendor().MLP
        self.pre_bbox_head = mlp(self.hidden_dim, self.hidden_dim, 4, 3)
        self.dec_bbox_head = nn.ModuleList([mlp(self.hidden_dim, self.hidden_dim, 4, 3) for _ in range(3)])
        import hashlib
        for name,head in [("pre_bbox_head",self.pre_bbox_head)]+[(f"dec_bbox_head.{i}",h) for i,h in enumerate(self.dec_bbox_head)]:
            for j,linear in enumerate(head.layers[:-1]):
                seed=int.from_bytes(hashlib.sha256(f"r0-initializer-v1/0/{name}/layers.{j}".encode()).digest()[:8],"big") & ((1<<63)-1)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed);linear.reset_parameters()
            nn.init.zeros_(head.layers[-1].weight);nn.init.zeros_(head.layers[-1].bias)
        self.register_buffer("r0_geometry_parameters",torch.tensor([1e-5,3.,300.,4.],dtype=torch.float64))
        self.trace_enabled=False
        self.record_microbatch=None

    def _get_encoder_input(self, *args, **kwargs):
        return _vendor().RTDETRTransformerv2._get_encoder_input(self, *args, **kwargs)

    def _generate_anchors(self, *args, **kwargs):
        return _vendor().RTDETRTransformerv2._generate_anchors(self, *args, **kwargs)

    def _get_decoder_input(self, *args, **kwargs):
        return _vendor().RTDETRTransformerv2._get_decoder_input(self, *args, **kwargs)

    def _select_topk(self, *args, **kwargs):
        return _vendor().RTDETRTransformerv2._select_topk(self, *args, **kwargs)

    def forward(self, feats, targets=None):
        memory, shapes = self._get_encoder_input(feats)
        vd = _vendor()
        if self.training and self.num_denoising > 0:
            noise_classes=[]
            handle=self.denoising_class_embed.register_forward_pre_hook(lambda module,args:noise_classes.append(args[0].detach().clone()))
            try:
                dn_logits,dn_boxes,mask,dn_meta=vd.get_contrastive_denoising_training_group(targets,self.num_classes,self.num_queries,self.denoising_class_embed,
                    num_denoising=self.num_denoising,label_noise_ratio=self.label_noise_ratio,box_noise_scale=self.box_noise_scale)
            finally:handle.remove()
        else:
            dn_logits,dn_boxes,mask,dn_meta=None,None,None,None;noise_classes=[]
        noise={"classes":noise_classes[0] if noise_classes else None,"boxes_unact":dn_boxes.detach().clone() if dn_boxes is not None else None,
               "attention_mask":mask.detach().clone() if mask is not None else None,"metadata":dn_meta,"embedding_calls":len(noise_classes)}
        content, init_unact, enc_boxes, enc_logits = self._get_decoder_input(memory, shapes, dn_logits, dn_boxes)
        valid = torch.ones(content.shape[:2], dtype=torch.bool, device=content.device)
        if dn_meta is not None:
            counts = [len(t["labels"]) for t in targets]
            max_gt = max(counts)
            dn_valid = torch.arange(max_gt, device=content.device)[None,:] < torch.tensor(counts, device=content.device)[:,None]
            valid[:, :dn_meta["dn_num_split"][0]] = dn_valid.tile((1, 2*dn_meta["dn_num_group"]))
        from ..localization.terminal_geometry import dn_roles
        expected,_=dn_roles(dn_meta,targets,self.num_queries,content.device) if targets is not None else (valid,None)
        if not torch.equal(valid,expected):raise ValueError("R0 DN role validity mismatch")
        ref = torch.sigmoid(init_unact.float())
        output = content
        previous_feature = 0
        layers = []
        from ..localization.consumer_geometry import layer as geometry_layer,tensor
        tensor("decoder/memory",memory);tensor("decoder/content",content)
        for index, layer in enumerate(self.decoder.layers):
            validate_reference(ref, valid)
            with geometry_layer(layer,index,ref,valid):
                query_pos = self.query_pos_head(ref)
                tensor("position/query_pos",query_pos)
                output = layer(output, ref.unsqueeze(2), memory, shapes, mask, None, query_pos)
                tensor("decoder/layer_features",output)
            require_finite(output, "decoder features")
            if index == 0:
                pre_box = torch.sigmoid(self.pre_bbox_head(output).float() + vd.inverse_sigmoid(ref).float())
                pre_score = self.dec_score_head[0](output)
                tensor("decoder/pre_box",pre_box);tensor("decoder/pre_score",pre_score)
                positive_box("R0/pre_box",pre_box,valid)
                initial = pre_box.detach()
            regression_reference=initial if index==0 else ref
            residual=self.dec_bbox_head[index](output+previous_feature).float()
            require_finite(residual,"R0 scalar regression residual")
            boxes=torch.sigmoid(residual+vd.inverse_sigmoid(regression_reference).float())
            try:
                positive_box("R0/decode/box32",boxes,valid)
            except Exception as error:
                if hasattr(error,"evidence"):
                    error.evidence.update(decoder_layer=index,microbatch=self.record_microbatch,representation="R0",dn_queries=dn_meta["dn_num_split"][0] if dn_meta else 0)
                    error.evidence["query_context"]=[{"sample":b,"query":q,"kind":"DN" if dn_meta and q<dn_meta["dn_num_split"][0] else "ordinary","attention_reference":ref[b,q].detach().cpu().tolist(),
                        "regression_reference":regression_reference[b,q].detach().cpu().tolist(),"image_id":targets[b].get("image_id") if targets else None} for b,q in error.evidence.get("bad_indices",[])[:16]]
                raise
            scores = self.dec_score_head[index](output)
            require_finite(scores, "classification logits")
            layers.append({"pred_boxes": boxes, "pred_logits": scores,
                           "attention_reference":ref.detach(),"regression_reference":regression_reference.detach(),"query_valid":valid})
            previous_feature = output.detach()
            ref = boxes.detach()

        normal, denoising = [], []
        dn_count = dn_meta["dn_num_split"][0] if dn_meta is not None else 0
        for layer in layers:
            normal.append({k: v[:, dn_count:] if isinstance(v, torch.Tensor) else v for k, v in layer.items()})
            if dn_meta is not None:
                denoising.append({k: v[:, :dn_count] if isinstance(v, torch.Tensor) else v for k, v in layer.items()})
        if not self.training and not self.trace_enabled:
            return {k: normal[-1][k] for k in ("pred_logits", "pred_boxes")}
        result = dict(normal[-1])
        result["pre_outputs"] = {"pred_logits": pre_score[:, dn_count:], "pred_boxes": pre_box[:, dn_count:]}
        if self.training:
            result["aux_outputs"]=[dict(n) for n in normal[:-1]]
            result["enc_aux_outputs"] = [{"pred_boxes": b, "pred_logits": s} for b, s in zip(enc_boxes, enc_logits)]
            result["enc_meta"] = {"class_agnostic": False}
            if dn_meta is not None:
                result["dn_outputs"]=[dict(n) for n in denoising]
                result["dn_pre_outputs"] = {"pred_logits": pre_score[:, :dn_count], "pred_boxes": pre_box[:, :dn_count]}
                result["dn_meta"] = dn_meta
            result["dn_noise"]=noise
        else:
            result["trace_outputs"] = normal
        return result


def build_r0(config=R0Config(), *, pretrained_path=None, pretrained_sha256=None, synthetic=False):
    """Always build on CPU; callers explicitly move it after resource admission."""
    config.validate(synthetic=synthetic)
    from ..baseline.training_v2b import V2BConfig, _build_vendor_objects, _configure_model_sampling, _load_pretrained
    root = Path(__file__).resolve().parents[3]
    base_config = V2BConfig(seed=config.seed, input_size=128, physical_batch_size=config.physical_batch_size,
                           accumulation_steps=2, amp_dtype=config.amp_dtype, pretrained_required=not synthetic,
                           sampling_backend=config.sampling_backend)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model, old_criterion, postprocessor, resolved, _, _ = _build_vendor_objects(root, base_config)
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model.decoder = R0Transformer(model.decoder)
    for component in (model.encoder, model.decoder):
        component.eval_spatial_size = list(config.input_hw)
    model.encoder.build_2d_sincos_position_embedding = row_major_position_embedding
    h, w = config.input_hw
    for index in model.encoder.use_encoder_idx:
        stride = model.encoder.feat_strides[index]
        setattr(model.encoder, f"pos_embed{index}", row_major_position_embedding(w//stride, h//stride, model.encoder.hidden_dim))
    model.decoder.anchors, model.decoder.valid_mask = model.decoder._generate_anchors()
    sampling = _configure_model_sampling(model, base_config)
    after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if after - before != 132612:
        raise RuntimeError("unexpected R0 trainable parameter delta")
    if pretrained_path is None and not synthetic:
        raise ValueError("formal R0 requires the verified ImageNet backbone")
    pretrained = None
    if pretrained_path is not None:
        pretrained = _load_pretrained(model, Path(pretrained_path), pretrained_sha256)
    from .criterion import R0Criterion
    criterion = R0Criterion(old_criterion.matcher, alpha=old_criterion.alpha, gamma=old_criterion.gamma)
    return model, criterion, {"config": config.binding(), "criterion":criterion.recipe(), "trainable_parameters": after,
                             "parameter_delta": after-before, "sampling": sampling, "pretrained": pretrained,
                             "attention_class": type(model.decoder.decoder.layers[0]).__module__ + "." + type(model.decoder.decoder.layers[0]).__name__}


class R0EMA:
    """Original EMA math, with immutable geometry copied and integer BN buffers unchanged."""
    def __init__(self, model, decay=.9999, warmups=2000):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay, self.warmups, self.updates = decay, warmups, 0

    @torch.no_grad()
    def update(self, model):
        import math
        self.updates += 1
        d = self.decay * (1. - math.exp(-self.updates / self.warmups))
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if name.rsplit(".", 1)[-1] in {"anchors", "valid_mask", "num_points_scale", "r0_geometry_parameters"}:
                if not torch.equal(value, source[name]):
                    raise RuntimeError("immutable model/EMA geometry differs: " + name)
            elif value.is_floating_point():
                value.mul_(d)
                value.add_((1.-d) * source[name].detach())

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates, "decay": self.decay, "warmups": self.warmups}

    def load_state_dict(self, state):
        if state["decay"] != self.decay or state["warmups"] != self.warmups:
            raise ValueError("EMA recipe mismatch")
        self.module.load_state_dict(state["module"], strict=True)
        self.updates = state["updates"]
