# in /workspace/projects/uncertainbird/configs/experiment/Wrappers/eat_dropout_hooks.py

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


def _add_tag(m: nn.Module, tag: str):
    tags = getattr(m, "_mcd_tags", [])
    tags.append(tag)
    setattr(m, "_mcd_tags", tags)

def _is_resblock_1d_tf(m: nn.Module) -> bool:
    return (
        m.__class__.__name__ == "ResBlock1dTF"
        and hasattr(m, "block_t")
        and hasattr(m, "block_f")
        and hasattr(m, "shortcut")
    )

def _is_down_block(m: nn.Module) -> bool:
    return m.__class__.__name__ == "Down" and hasattr(m, "down")

def _is_taggregate(obj: object) -> bool:
    return (
        getattr(obj, "__class__", type("X",(object,),{})).__name__ == "TAggregate"
        and hasattr(obj, "transformer_enc")
        and hasattr(obj, "fc")
    )

def attach_eat_dropout_hooks_fine(
    model: nn.Module,
    *,
    p_conv_res: float = 0.0,   # after every Conv1d in ResBlock1dTF
    p_conv_down: float = 0.00, # after Conv1d in each Down
    p_project: float = 0.0,    # after project Conv1d
    p_token: float = 0.0,      # token dropout before TransformerEncoder
    p_head: float = 0.0        # dropout before tf.fc
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    mc_flag = {"on": False}
    setattr(model, "_eat_mc_flag", mc_flag)

    def make_conv1d_out_hook(p: float):
        def hook(mod: nn.Module, inputs, output):
            training = mod.training or mc_flag["on"]
            if p > 0.0 and isinstance(output, torch.Tensor) and output.dim() == 3:
                return F.dropout1d(output, p=p, training=training)
            return output
        return hook

    # 1) every Conv1d inside ResBlock1dTF.block_t / block_f
    for m in model.modules():
        if _is_resblock_1d_tf(m):
            for sub in m.block_t.modules():
                if isinstance(sub, nn.Conv1d):
                    h = sub.register_forward_hook(make_conv1d_out_hook(p_conv_res))
                    handles.append(h)
                    _add_tag(sub, f"HOOK Dropout1d(p={p_conv_res}) @ ResBlock1dTF.block_t")
            for sub in m.block_f.modules():
                if isinstance(sub, nn.Conv1d):
                    h = sub.register_forward_hook(make_conv1d_out_hook(p_conv_res))
                    handles.append(h)
                    _add_tag(sub, f"HOOK Dropout1d(p={p_conv_res}) @ ResBlock1dTF.block_f")

    # 2) main Conv1d inside each Down.down
    for m in model.modules():
        if _is_down_block(m):
            for sub in m.down.modules():
                if isinstance(sub, nn.Conv1d):
                    h = sub.register_forward_hook(make_conv1d_out_hook(p_conv_down))
                    handles.append(h)
                    _add_tag(sub, f"HOOK Dropout1d(p={p_conv_down}) @ Down.down")
                    break  # just the first/main conv

    # 3) project conv
    if hasattr(model, "project") and isinstance(model.project, nn.Conv1d):
        h = model.project.register_forward_hook(make_conv1d_out_hook(p_project))
        handles.append(h)
        _add_tag(model.project, f"HOOK Dropout1d(p={p_project}) @ project")

    # 4) token dropout before transformer
    tf = getattr(model, "tf", None)
    if tf is not None and _is_taggregate(tf) and hasattr(tf, "transformer_enc"):
        def token_pre_hook(mod: nn.Module, inputs):
            (x,) = inputs
            training = tf.training or mc_flag["on"]
            if p_token > 0.0:
                x = F.dropout(x, p=p_token, training=training)
            return (x,)
        handles.append(tf.transformer_enc.register_forward_pre_hook(token_pre_hook))
        _add_tag(tf.transformer_enc, f"PRE-HOOK TokenDropout(p={p_token})")

    # 5) head dropout before fc
    if tf is not None and hasattr(tf, "fc") and isinstance(tf.fc, nn.Linear):
        def head_pre_hook(mod: nn.Module, inputs):
            (x,) = inputs
            training = tf.training or mc_flag["on"]
            if p_head > 0.0:
                x = F.dropout(x, p=p_head, training=training)
            return (x,)
        handles.append(tf.fc.register_forward_pre_hook(head_pre_hook))
        _add_tag(tf.fc, f"PRE-HOOK HeadDropout(p={p_head})")

    setattr(model, "_mcd_handles", handles)
    return handles


def set_eat_mc_mode(model: nn.Module, enabled: bool = True, freeze_batchnorm: bool = True) -> None:
    """Enable/disable MC-dropout. Keeps BN layers in eval to avoid stat drift."""
    if not hasattr(model, "_eat_mc_flag"):
        return
    model._eat_mc_flag["on"] = bool(enabled)
    if freeze_batchnorm:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()


def describe_eat_setup(model: nn.Module, max_len: int = 72) -> None:
    """
    Pretty-print what EAT-style dropout hooks were attached.
    Looks for:
      - model._eat_mc_flag["on"]          (enable/disable state)
      - model._mcd_handles                (list of RemovableHandle)
      - submodule._mcd_tags               (list of string tags added by _add_tag)
    """
    enabled = None
    flag = getattr(model, "_eat_mc_flag", None)
    if isinstance(flag, dict):
        enabled = flag.get("on", None)

    handles = getattr(model, "_mcd_handles", [])
    print("\n=== EAT MC hooks ===")
    print(f"enabled: {enabled}")
    print(f"#handles: {len(handles)}")

    # collect tagged modules
    rows = []
    kinds = []
    for name, m in model.named_modules():
        tags = getattr(m, "_mcd_tags", None)
        if not tags:
            continue
        for tag in tags:
            # tag examples from your code:
            #  "HOOK Dropout1d(p=0.1) @ ResBlock1dTF.block_t"
            #  "PRE-HOOK TokenDropout(p=0.0)"
            kind = "pre" if tag.startswith("PRE-HOOK") else "post"
            kinds.append(kind)
            t = tag if len(tag) <= max_len else "…" + tag[-(max_len-1):]
            rows.append((name, t, kind))

    if not rows:
        print("(no tagged hook locations found)")
    else:
        # summary
        c = Counter(kinds)
        if c:
            print("summary:", ", ".join(f"{k}={v}" for k, v in c.items()))
        # detailed list
        for name, tag, kind in rows:
            print(f"{name:<{max_len}} :: kind={kind}  {tag}")

    print("=== end ===\n")