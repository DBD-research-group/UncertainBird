# convnext_eat_hooks_for_your_model.py
import torch.nn as nn
import torch.nn.functional as F

class _OutDropHook:
    def __init__(self, p: float, spatial: bool, tag: str):
        self.p = float(p)
        self.spatial = bool(spatial)
        self.tag = tag
        self.enabled = True
    def __call__(self, module, args, out):
        if not self.enabled or self.p <= 0.0:
            return out
        if self.spatial and out.dim() >= 4:
            return F.dropout2d(out, p=self.p, training=True)
        return F.dropout(out, p=self.p, training=True)

def attach_convnext_eat_hooks(
    hf_convnext_for_cls: nn.Module,
    p_stem: float = 0.05,
    p_block: float = 0.05,
    p_head: float = 0.10,
):
    """Attach EAT-style MC dropout hooks tailored to your printed structure."""
    # cleanup if re-attaching
    for h in getattr(hf_convnext_for_cls, "_mcd_handles", []):
        try: h.remove()
        except: pass
    hf_convnext_for_cls._mcd_handles = []
    hf_convnext_for_cls._mcd_hooks = []
    hf_convnext_for_cls._mcd_tags = []

    # 1) STEM: convnext.embeddings.patch_embeddings (Conv2d)
    stem = hf_convnext_for_cls.convnext.embeddings.patch_embeddings
    if isinstance(stem, nn.Conv2d) and p_stem > 0:
        hk = _OutDropHook(p_stem, spatial=True, tag="stem:embeddings.patch_embeddings")
        handle = stem.register_forward_hook(hk)
        hf_convnext_for_cls._mcd_handles.append(handle)
        hf_convnext_for_cls._mcd_hooks.append(hk)
        hf_convnext_for_cls._mcd_tags.append(hk.tag)

    # 2) BLOCKS: every ConvNextLayer under convnext.encoder.stages.*.layers.*
    if p_block > 0:
        for stage_idx, stage in enumerate(hf_convnext_for_cls.convnext.encoder.stages):
            for layer_idx, layer in enumerate(stage.layers):
                if layer.__class__.__name__.lower().startswith("convnextlayer"):
                    hk = _OutDropHook(p_block, spatial=True, tag=f"block:s{stage_idx}.l{layer_idx}")
                    handle = layer.register_forward_hook(hk)
                    hf_convnext_for_cls._mcd_handles.append(handle)
                    hf_convnext_for_cls._mcd_hooks.append(hk)
                    hf_convnext_for_cls._mcd_tags.append(hk.tag)

    # 3) HEAD: before classifier Linear(1024->21)
    if isinstance(hf_convnext_for_cls.classifier, nn.Linear) and p_head > 0:
        # Easiest: wrap with Dropout -> Linear so inference graph includes dropout
        lin = hf_convnext_for_cls.classifier
        hf_convnext_for_cls.classifier = nn.Sequential(nn.Dropout(p_head), lin)
        # also add a hook on classifier for robustness
        hk = _OutDropHook(p_head, spatial=False, tag="head:classifier")
        handle = hf_convnext_for_cls.classifier.register_forward_hook(hk)
        hf_convnext_for_cls._mcd_handles.append(handle)
        hf_convnext_for_cls._mcd_hooks.append(hk)
        hf_convnext_for_cls._mcd_tags.append(hk.tag)

    hf_convnext_for_cls._mcd_enabled = True

def set_convnext_eat_mc_mode(hf_convnext_for_cls: nn.Module, enabled: bool = True):
    hf_convnext_for_cls.eval()  # keep LayerNorm deterministic
    for hk in getattr(hf_convnext_for_cls, "_mcd_hooks", []):
        hk.enabled = bool(enabled)
    hf_convnext_for_cls._mcd_enabled = bool(enabled)

def describe_convnext_eat_setup(hf_convnext_for_cls: nn.Module, max_len=72):
    print("\n=== ConvNeXt EAT-style MC setup ===")
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
