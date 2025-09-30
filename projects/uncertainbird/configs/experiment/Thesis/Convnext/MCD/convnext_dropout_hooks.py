# convnext_dropout_hooks.py
import torch.nn as nn
import torch.nn.functional as F
# convnext_hf_dropout_hooks.py
import torch.nn as nn
import torch.nn.functional as F

class _OutDropHook:
    def __init__(self, p: float, spatial: bool, tag: str):
        self.p = float(p); self.spatial = bool(spatial); self.tag = tag; self.enabled = True
    def __call__(self, module, args, out):
        if not self.enabled or self.p <= 0.0:
            return out
        if self.spatial and hasattr(out, "dim") and out.dim() >= 4:
            return F.dropout2d(out, p=self.p, training=True)
        return F.dropout(out, p=self.p, training=True)

class _InDropPreHook:
    def __init__(self, p: float, tag: str):
        self.p = float(p)
        self.tag = tag
        self.enabled = True
        self.spatial = False   # <- add this line so describe() can read it
    def __call__(self, module, inputs):
        if not self.enabled or self.p <= 0.0 or len(inputs) == 0:
            return
        x = inputs[0]
        return (F.dropout(x, p=self.p, training=True),)

def _clear_old(m: nn.Module):
    for h in getattr(m, "_mcd_handles", []):
        try: h.remove()
        except: pass
    m._mcd_handles, m._mcd_hooks, m._mcd_tags = [], [], []
    setattr(m, "_mcd_enabled", False)

def _add(m, handle, hook, tag):
    m._mcd_handles.append(handle); m._mcd_hooks.append(hook); m._mcd_tags.append(tag)

def attach_convnext_hooks(hf_model: nn.Module, p_stem=0.05, p_block=0.05, p_head=0.10):
    """
    Matches your printout:
      model.convnext.embeddings.patch_embeddings
      model.convnext.encoder.stages[i].layers[j]
      model.classifier (Linear)
    """
    _clear_old(hf_model)

    # --- STEM ---
    try:
        stem = hf_model.convnext.embeddings.patch_embeddings   # Conv2d
        if isinstance(stem, nn.Conv2d) and p_stem > 0:
            hk = _OutDropHook(p_stem, spatial=True, tag="stem:embeddings.patch_embeddings")
            h = stem.register_forward_hook(hk); _add(hf_model, h, hk, hk.tag)
    except Exception:
        pass

    # --- BLOCKS ---
    if p_block > 0:
        try:
            for si, stage in enumerate(hf_model.convnext.encoder.stages):
                for li, layer in enumerate(stage.layers):       # ConvNextLayer
                    hk = _OutDropHook(p_block, spatial=True, tag=f"block:s{si}.l{li}")
                    h = layer.register_forward_hook(hk); _add(hf_model, h, hk, hk.tag)
        except Exception:
            pass

    # --- HEAD (Linear) ---
    clf = getattr(hf_model, "classifier", None)
    if isinstance(clf, nn.Linear) and p_head > 0:
        hk = _InDropPreHook(p_head, tag="head:classifier_pre")
        h = clf.register_forward_pre_hook(hk); _add(hf_model, h, hk, hk.tag)

    setattr(hf_model, "_mcd_enabled", True)

def set_convnext_mc_mode(hf_model: nn.Module, enabled: bool = True):
    for hk in getattr(hf_model, "_mcd_hooks", []):
        hk.enabled = bool(enabled)
    setattr(hf_model, "_mcd_enabled", bool(enabled))




def describe_modules(model, limit=200, show_params=True):
    def nparams(mod): return sum(p.numel() for p in mod.parameters(recurse=False))
    for i, (name, m) in enumerate(model.named_modules()):
        line = f"[{i:03d}] {name:40s} {m.__class__.__name__}"
        if show_params: line += f" | params={nparams(m)}"
        print(line)
        if i+1 >= limit: break

def describe_convnext_setup(hf_convnext_for_cls: nn.Module, max_len=72):
    describe_modules(hf_convnext_for_cls)
    print("\n=== ConvNeXt MC setup ===")
    print("enabled:", getattr(hf_convnext_for_cls, "_mcd_enabled", None))
    tags = getattr(hf_convnext_for_cls, "_mcd_tags", [])
    hooks = getattr(hf_convnext_for_cls, "_mcd_hooks", [])
    if not tags:
        print("(no hooks registered)")
    else:
        for tag, hk in zip(tags, hooks):
            t = tag if len(tag) <= max_len else "…" + tag[-(max_len-1):]
            print(f"{t:<{max_len}} :: p={hk.p} spatial={hk.spatial} enabled={hk.enabled}")
    print("=== end ===\n")
