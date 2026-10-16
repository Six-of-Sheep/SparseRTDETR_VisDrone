"""No-pickle fixed64 cache contract, independent of model execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np

from ..data_protocol.schema import stable_annotation_id
from .operators import physical_support, weighting_function
from .distribution import NUMERICAL_IMPLEMENTATION
from .consumer_geometry import RULES

SCHEMA = "f0-e120-ema-fixed64-cache-consumer-geometry-v3"
SHAPES = {"boxes": (3,300,4), "references": (3,300,4), "class_logits": (3,300,10),
          "effective_logits": (3,300,4,33), "probabilities": (3,300,4,33),
          "teacher_probabilities": (300,4,33)}


def validate_cache(cache, row, *, checkpoint_sha256, selection_sha256, raw_target=None):
    expected = {"schema": SCHEMA, "arm": "F0", "seed": 0, "epoch": 120, "ema": True,
                "numerical_implementation":NUMERICAL_IMPLEMENTATION,"geometry_contract":RULES["name"],
                "image_id": row["stable_image_id"], "image_sha256": row["image_sha256"],
                "annotation_sha256": row["annotation_sha256"], "original_hw": [row["height"],row["width"]],
                "input_hw": [768,1344], "checkpoint_sha256": checkpoint_sha256,
                "selection_sha256": selection_sha256}
    if set(cache) != set(expected) | set(SHAPES) or any(cache.get(k) != v for k,v in expected.items()):
        raise ValueError("trace cache identity/endpoint/schema mismatch")
    for name, shape in SHAPES.items():
        value = cache[name]
        if not isinstance(value,np.ndarray) or value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid trace tensor: " + name)
    for name in ("boxes", "references"):
        if np.any(cache[name][...,2:] <= 0):
            raise ValueError("trace nonpositive/unresolved geometry: " + name)
    if not np.array_equal(cache["references"][0],cache["references"][1]) or not np.array_equal(cache["references"][0],cache["references"][2]):
        raise ValueError("F0 trace support changed between layers")
    for name, logits, temperature in (("probabilities",cache["effective_logits"],1.),
                                     ("teacher_probabilities",cache["effective_logits"][-1],5.)):
        p = cache[name].astype(np.float64)
        z = logits.astype(np.float64)/temperature
        exponent = np.exp(z-z.max(-1,keepdims=True)); reference = exponent/exponent.sum(-1,keepdims=True)
        if np.any(p<0) or np.max(np.abs(p.sum(-1)-1.))>3e-6 or np.max(np.abs(p-reference))>3e-6:
            raise ValueError("trace probability/logit or mass mismatch: " + name)
    physical_support(cache["references"], ascending=True) # original independent resolution gate
    p=cache["probabilities"].astype(np.float64)
    relative=(p*(.5+weighting_function()/4.)).sum(-1)/p.sum(-1)
    ref=cache["references"].astype(np.float64)
    delta=np.array([-1.,-1.,1.,1.])*ref[...,[2,3,2,3]]*relative
    decoded=np.concatenate((ref[...,:2]+(delta[...,:2]+delta[...,2:])/2.,delta[...,2:]-delta[...,:2]),-1)
    # This is a saved-FP32 versus FP64 analytic check, not cross-machine bit identity.
    expected_fp32=decoded.astype(np.float32)
    ulp=np.abs(np.spacing(expected_fp32)).astype(np.float64)
    if np.any(np.abs(expected_fp32.astype(np.float64)-cache["boxes"].astype(np.float64)) > 3*ulp):
        raise ValueError("trace decoded boxes differ from saved effective probabilities")
    if raw_target is not None:
        rows = raw_target["rows"]
        if len(rows) != row["raw_annotation_rows"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("trace raw row count/identity mismatch")
        if any(r["id"] != stable_annotation_id(row["stable_image_id"],r["physical_line"],r["raw_line_sha256"]) for r in rows):
            raise ValueError("trace raw annotation belongs to another image")
    return {"image_id":row["stable_image_id"],"shapes_and_finite":True,"logit_probability_identity":True,
            "decoded_geometry":True,"raw_identity":raw_target is not None}


def validate_collection_identities(metadata, selected, *, checkpoint_sha256, selection_sha256):
    expected_ids = [r["stable_image_id"] for r in selected]
    ids = [r["image_id"] for r in metadata]
    if len(expected_ids)!=64 or len(set(expected_ids))!=64 or len(ids)!=64 or len(set(ids))!=64 or ids!=expected_ids:
        raise ValueError("trace must contain the complete ordered unique fixed64 set")
    if any(m["checkpoint_sha256"]!=checkpoint_sha256 or m["selection_sha256"]!=selection_sha256 for m in metadata):
        raise ValueError("mixed trace endpoint or selection")
    return True


def save_cache(path, cache):
    path = Path(path); partial=path.with_suffix(path.suffix+".partial")
    if path.exists() or partial.exists(): raise FileExistsError("trace cache is immutable")
    metadata = {k:v for k,v in cache.items() if k not in SHAPES}
    with partial.open("xb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False)),
                            **{k:cache[k] for k in SHAPES})
        stream.flush();os.fsync(stream.fileno())
    os.replace(partial,path)


def read_cache(path):
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=set(SHAPES)|{"metadata"}: raise ValueError("invalid trace archive keys")
        result=json.loads(str(data["metadata"].item()))
        if set(result)&set(SHAPES): raise ValueError("trace metadata shadows an array")
        result.update({k:data[k].copy() for k in SHAPES})
    return result
