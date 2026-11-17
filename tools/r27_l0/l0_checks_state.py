"""R27-002 L0 checks: helpers shared by the cpu / zero / accept stages.  No model forward happens in this module.

``initial_vs_r27`` compares a freshly constructed L0 engine (before any update) with the original R27 formal-P e060
state: the native part must be identical tensor by tensor; what R27 alone carries is listed as declared differences;
anything else stops the stage.
"""
import json
import sys

import l0_common as K

sys.path.insert(0, str(K.ROOT / "tools" / "r27"))  # r27_run: only micro_steps (pure) and native_l (reference constructor)

SIDE_KEYS = {"status", "schema", "scope", "arm", "contract_sha256", "identity_sha256", "parent_sha256", "global_update",
             "cursor", "config", "audit", "role", "activation_checkpoint", "torch", "file"}


def first_line(exc):
    return (str(exc) or repr(exc)).strip().splitlines()[0][:400]


def refused(case, fn):
    try:
        fn()
    except (ValueError, RuntimeError, KeyError) as exc:
        return {"case": case, "error_type": type(exc).__name__, "error": first_line(exc)}
    raise AssertionError("accepted: " + case)


def r27_rows(c, arm, n):
    rows = []
    with K.pinned(c["R27_updates"][arm]).open() as f:
        for _ in range(n):
            rows.append(json.loads(f.readline()))
    K.require([r["input"]["update"] for r in rows] == [K.PARENT_UPDATE + 1 + i for i in range(n)], "R27 UPDATES first rows")
    return rows


def bn_digest(model):
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    items = [(n, v) for n, v in model.state_dict().items() if n.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    return {"buffers": len(items), "sha256": tensor_digest(items)}


def r27_only(name):
    """Names that exist only because R27 is installed (local shallow backbone/projection and the decoder local read)."""
    from sparse_rtdetr.research import r27_local_memory as R
    return R.is_new(name) or name == R.LOCAL or name.startswith(R.LOCAL + ".") or name.endswith("." + R.READ)


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and bool(K.torch_mod().equal(a, b))


def _native_map(mine, theirs, label, eq, same, res):
    extra = [k for k in theirs if k not in mine]
    K.require(all(k in theirs for k in mine), f"{label}: native entries missing in the R27 e060 state")
    K.require(all(r27_only(k) for k in extra), (f"{label}: undeclared extra R27 entries", [k for k in extra if not r27_only(k)][:5]))
    bad = [k for k in mine if not same(mine[k], theirs[k])]
    K.require(not bad, (f"{label}: native entries differ from the R27 e060 state", len(bad), bad[:5]))
    res["equal"][eq] = len(mine)
    res["native_order_equal"][eq] = [k for k in theirs if k in mine] == list(mine)
    if extra:
        res["declared_differences"].append({"where": label, "kind": "entries only in R27-P", "count": len(extra),
                                            "first": extra[0], "last": extra[-1]})


def _by_name(opt):
    out = {}
    for g in opt["param_groups"]:
        for n, i in zip(g["param_names"], g["params"]):
            K.require(n not in out, "duplicate optimizer parameter name")
            out[n] = opt["state"].get(i, {})
    return out


def initial_vs_r27(c, e):
    from formal_data import cursor
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    K.require(e.completed_updates == K.PARENT_UPDATE and not e.phase, "initial comparison needs the untouched L0 start state")
    mine = S.capture(e)
    path = K.pinned(c["R27_P_e060"])
    side = K.read_json(K.pinned(c["R27_P_e060_side"]))
    K.require(set(side) == SIDE_KEYS and side["file"] == c["R27_P_e060"], "R27 e060 side-car keys / file identity")
    theirs = torch.load(path, map_location="cpu", weights_only=True)
    res = {"R27_state": c["R27_P_e060"], "equal": {}, "native_order_equal": {}, "declared_differences": []}
    dd = res["declared_differences"]
    K.require(set(theirs) == set(mine) | {"r27_binding_sha256"}, ("top-level keys", sorted(set(theirs) ^ set(mine))))
    dd.append({"where": "state top level", "kind": "R27 binding digest (L0 saves carry l0_binding_sha256 instead)",
               "key": "r27_binding_sha256"})
    for k in ("version", "parent_sha256", "config", "activation_checkpoint"):
        K.require(type(mine[k]) is type(theirs[k]) and mine[k] == theirs[k], f"top-level field differs: {k}")
    res["equal"]["top_level_fields"] = 4
    K.require((mine["arm"], theirs["arm"]) == ("L0", "R27-P"), "arm names")
    dd.append({"where": "state top level", "kind": "arm name", "L0": mine["arm"], "R27": theirs["arm"]})
    _native_map(mine["model"], theirs["model"], "online model state_dict", "online_model_tensors", _same, res)
    _native_map(mine["ema"]["module"], theirs["ema"]["module"], "EMA module state_dict", "EMA_module_tensors", _same, res)
    scalars = sorted(k for k in mine["ema"] if k != "module")
    K.require(set(theirs["ema"]) == set(mine["ema"]) and all(mine["ema"][k] == theirs["ema"][k] for k in scalars), "EMA scalars")
    res["equal"]["EMA_scalars"] = {k: mine["ema"][k] for k in scalars}
    _native_map(mine["model_modes"], theirs["model_modes"], "module training modes", "module_modes", lambda a, b: a is b, res)
    # optimizer: per-parameter state by name, group settings, group membership
    mo, to = _by_name(mine["optimizer"]), _by_name(theirs["optimizer"])
    _native_map(mo, to, "optimizer per-parameter state", "optimizer_parameter_states", ck.exact_equal, res)
    res["equal"]["optimizer_state_tensors"] = sum(isinstance(v, torch.Tensor) for s in mo.values() for v in s.values())
    res["optimizer_state_fields"] = sorted({k for s in mo.values() for k in s})
    mg, tg = mine["optimizer"]["param_groups"], theirs["optimizer"]["param_groups"]
    K.require(set(mine["optimizer"]) == set(theirs["optimizer"]) == {"state", "param_groups"} and len(mg) == len(tg), "optimizer layout")
    for g, og, names, onames in zip(mg, tg, mine["optimizer_names"], theirs["optimizer_names"]):
        K.require(set(g) == set(og), "optimizer group keys")
        for k in g:
            if k not in ("params", "param_names"):
                K.require(type(g[k]) is type(og[k]) and g[k] == og[k], f"optimizer group {g['name']} setting differs: {k}")
        K.require(g["param_names"] == names == [n for n in onames if not r27_only(n)] and og["param_names"] == onames,
                  f"optimizer group {g['name']} native membership/order differs")
        extra = [n for n in onames if r27_only(n)]
        dd.append({"where": f"optimizer group {g['name']}", "kind": "members only in R27-P", "count": len(extra),
                   "native_members": len(names)})
    res["equal"]["optimizer_groups"] = [{k: v for k, v in g.items() if k not in ("params", "param_names")} for g in mg]
    # engine clock / per-parameter step counters
    me, te = mine["engine"], theirs["engine"]
    K.require(set(me) == set(te), "engine state keys")
    for k in me:
        if k != "parameter_optimizer_steps":
            K.require(type(me[k]) is type(te[k]) and me[k] == te[k], f"engine field differs: {k}")
    _native_map(me["parameter_optimizer_steps"], te["parameter_optimizer_steps"], "engine parameter_steps",
                "parameter_steps", lambda a, b: type(a) is type(b) and a == b, res)
    res["equal"]["engine_fields"] = len(me) - 1
    res["equal"]["completed_updates"] = me["completed_updates"]
    K.require(me["completed_updates"] == mine["ema"]["updates"] == side["global_update"] == K.PARENT_UPDATE, "clocks")
    # RNG, cursor and the comparable side-car facts
    K.require(set(mine["rng"]) == set(theirs["rng"]), "RNG keys")
    for k in mine["rng"]:
        K.require(ck.exact_equal(mine["rng"][k], theirs["rng"][k]), f"RNG differs: {k}")
    res["equal"]["rng_components"] = sorted(mine["rng"])
    facts = {"cursor": cursor(e.completed_updates), "config": e.config.binding(), "parent_sha256": c["parent"]["sha256"],
             "activation_checkpoint": e.s15_activation_checkpoint, "torch": str(torch.__version__),
             "global_update": e.completed_updates}
    for k, v in facts.items():
        K.require(type(v) is type(side[k]) and v == side[k], f"side-car fact differs: {k}")
    res["equal"]["side_car_facts"] = sorted(facts)
    res["cursor"] = facts["cursor"]
    for k, v in (("schema", K.SCHEMA), ("arm", K.ARM), ("contract_sha256", c["contract_sha256"]),
                 ("identity_sha256", c["identity_sha256"])):
        K.require(side[k] != v, f"side-car field unexpectedly equal: {k}")
        dd.append({"where": "state side-car", "kind": k, "L0": v, "R27": side[k]})
    dd.append({"where": "state side-car", "kind": "scope / status / file / audit / role describe the saved file, not the state",
               "R27_scope": side["scope"]})
    res["undeclared_differences"] = []
    return res
