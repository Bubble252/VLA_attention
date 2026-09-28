"""Native OFT continuous L1 forward, matching the pinned upstream indexing.

Imports no model at module load. Callers own checkpoint/model/processor audit.
"""
from contextlib import nullcontext


def forward_l1(policy, action_head, proprio_projector, batch, *, device, visual_token_count):
    import torch
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
    device = torch.device(device)
    floating_dtype = next(action_head.parameters()).dtype
    labels = batch['labels'].to(device)
    inputs = {k: batch[k].to(device) for k in ('input_ids', 'attention_mask')}
    pixels = batch['pixel_values'].to(device=device, dtype=floating_dtype)
    proprio = batch['proprio'].reshape(len(labels), -1).to(device=device,dtype=floating_dtype)
    context = torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with context:
        output = policy(**inputs,pixel_values=pixels,labels=labels,proprio=proprio,
                        proprio_projector=proprio_projector,use_film=False,
                        output_hidden_states=True,output_projector_features=True,use_cache=False)
        selected = get_current_action_mask(labels[:,1:]) | get_next_actions_mask(labels[:,1:])
        if not (selected.sum(1)==NUM_ACTIONS_CHUNK*ACTION_DIM).all():
            raise ValueError('Truncated/malformed action token block')
        # Same shifted hidden-state slice as upstream run_forward_pass;
        # one proprio embedding is appended to the visual block.
        hidden = output.hidden_states[-1][:,visual_token_count+1:-1]
        if hidden.shape[:2] != selected.shape:
            raise ValueError('Visual/proprio prefix does not match labels')
        action_features = hidden[selected].reshape(len(labels),NUM_ACTIONS_CHUNK*ACTION_DIM,-1)
        prediction = action_head.predict_action(action_features.to(floating_dtype))
        target = batch['actions'].to(device=device,dtype=prediction.dtype)
        if prediction.shape != target.shape or tuple(target.shape[1:])!=(NUM_ACTIONS_CHUNK,ACTION_DIM):
            raise ValueError('Native chunk/action shape mismatch')
        loss = torch.nn.functional.l1_loss(prediction,target)
    if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite native OFT forward')
    return {'loss':loss,'prediction':prediction,'visual_and_proprio_features':output.projector_features,
            'action_hidden_states':action_features}
