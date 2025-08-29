# utils/resolve_models.py
from __future__ import annotations
import torch.nn as nn

def _looks_like_eat_soundnet_root(m: nn.Module) -> bool:
    """
    Heuristics for your SoundNet-EAT backbone:
      - has attr 'tf' with 'transformer_enc' and 'fc'
      - often has 'project' (Conv1d)
      - contains ResBlock1dTF / Down submodules
    """
    tf = getattr(m, "tf", None)
    has_tf = (tf is not None) and hasattr(tf, "transformer_enc") and hasattr(tf, "fc")
    has_project = hasattr(m, "project")
    has_res_or_down = False
    for sub in m.modules():
        cls = sub.__class__.__name__
        if cls in ("ResBlock1dTF", "Down"):
            has_res_or_down = True
            break
    return bool(has_tf and (has_project or has_res_or_down))

def resolve_soundnet_eat(root: nn.Module) -> nn.Module:
    """
    Return the submodule to which EAT hooks should be attached.
    Tries:
      1) root itself
      2) root.model (common wrapper pattern)
      3) search children for the first module that matches EAT-SoundNet signatures
    Raises if none found.
    """
    # 1) root itself
    if _looks_like_eat_soundnet_root(root):
        return root

    # 2) common wrapper field
    inner = getattr(root, "model", None)
    if inner is not None and _looks_like_eat_soundnet_root(inner):
        return inner

    # 3) fallback: scan descendants
    for mod in root.modules():
        if _looks_like_eat_soundnet_root(mod):
            return mod

    raise RuntimeError("Could not find SoundNet-EAT backbone (no module matches EAT signatures)")

# Optional: debug helper that also returns a dotted path
def resolve_soundnet_eat_with_path(root: nn.Module) -> tuple[nn.Module, str]:
    if _looks_like_eat_soundnet_root(root):
        return root, "<root>"

    inner = getattr(root, "model", None)
    if inner is not None and _looks_like_eat_soundnet_root(inner):
        return inner, "model"

    # walk with names to get a path
    for name, mod in root.named_modules():
        if _looks_like_eat_soundnet_root(mod):
            return mod, name or "<root>"

    raise RuntimeError("Could not find SoundNet-EAT backbone (no module matches EAT signatures)")
