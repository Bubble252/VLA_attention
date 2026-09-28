"""Deterministic fingerprints for a real optimizer continuation check."""
import hashlib
import json


def parameter_digest(modules):
    digest=hashlib.sha256()
    for module_name,module in modules:
        for name,param in module.named_parameters():
            if not param.requires_grad:continue
            tensor=param.detach().cpu().contiguous()
            digest.update(json.dumps([module_name,name,list(tensor.shape),str(tensor.dtype)]).encode())
            digest.update(tensor.view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_probe(modules,optimizer,scheduler,loss_fn):
    import torch
    for _,module in modules:module.train()
    params=[p for _,m in modules for p in m.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    loss=loss_fn()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite continuation loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    return {'loss':float(loss.detach()),'grad_norm':float(norm),'lr':scheduler.get_last_lr(),
            'parameter_sha256':parameter_digest(modules),'scheduler_epoch':scheduler.last_epoch}
