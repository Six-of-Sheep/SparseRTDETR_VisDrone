"""Peek at a raw-query cache so the analysis tool reads the right keys and the right image order.

usage: peek_raw_cache.py <cache.pth> [<cache.pth> ...]
"""
import sys

import torch

for path in sys.argv[1:]:
    print("=" * 20, path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print("type:", type(obj))
    if not isinstance(obj, dict):
        print("shape:", getattr(obj, "shape", None))
        continue
    print("keys:", list(obj.keys()))
    for key, value in obj.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            print("  %-16s tensor %s %s" % (key, tuple(shape), getattr(value, "dtype", "")))
        elif isinstance(value, (list, tuple)):
            head = []
            for item in value[:3]:
                head.append(item if not hasattr(item, "shape") else tuple(item.shape))
            print("  %-16s %s len=%d head=%s" % (key, type(value).__name__, len(value), head))
        else:
            print("  %-16s %r" % (key, value))
    for key in ("schema", "epoch", "EMA", "arm"):
        if key in obj:
            print("  %s = %r" % (key, obj[key]))
