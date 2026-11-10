"""Model adapter. Importing this module does not load a model."""
import hashlib

import numpy as np


def prepared_prompt(tokenizer, raw):
    if not tokenizer.chat_template:
        raise ValueError('A chat template is required')
    return tokenizer.apply_chat_template([{'role': 'user', 'content': raw}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def completion_id(tokenizer, prompt, label):
    base = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + label, add_special_tokens=False).input_ids
    if full[:len(base)] != base or len(full) != len(base) + 1:
        raise ValueError(f'{label!r} is not a single appended token in this context')
    return int(full[-1])


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def forward_with_hook(model, enc, replacement=None, block_index=26):
    import torch
    calls, outputs = [], []
    def hook(_module, _inputs, output):
        calls.append(1)
        tensor = output[0] if isinstance(output, tuple) else output
        outputs.append(tensor[0, -1, :].detach().float().cpu().numpy())
        if replacement is None:
            return output
        changed = tensor.clone()
        changed[0, -1, :] = torch.as_tensor(replacement, device=tensor.device, dtype=tensor.dtype)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed
    handle = model.model.layers[block_index].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            result = model(**enc, output_hidden_states=True, return_dict=True)
    finally:
        handle.remove()
    if len(calls) != 1:
        raise RuntimeError(f'Expected exactly one hook, got {len(calls)}')
    state = result.hidden_states[block_index + 1][0, -1, :].detach().float().cpu().numpy()
    return result.logits[0, -1].detach().float().cpu().numpy(), state, outputs[0]


def sanity_check(model, enc, ids, cached, state_atol=0.01, logits_atol=0.01):
    baseline, state, hooked = forward_with_hook(model, enc)
    if not np.allclose(state, hooked, atol=state_atol, rtol=0):
        raise AssertionError('Hook output differs from hidden-state index 27')
    if not np.allclose(cached, state, atol=state_atol, rtol=0):
        raise AssertionError('Cached and live activations differ')
    identity, _, _ = forward_with_hook(model, enc, hooked)
    zero, _, _ = forward_with_hook(model, enc, hooked + np.zeros_like(hooked))
    for name, logits in [('identity', identity), ('zero', zero)]:
        if not np.allclose(logits[list(ids)], baseline[list(ids)], atol=logits_atol, rtol=0):
            raise AssertionError(f'{name} logits differ from baseline')
    return {'hook_count': 1, 'hidden_hook_max_abs': float(np.max(np.abs(state-hooked))),
            'cached_live_max_abs': float(np.max(np.abs(cached-state))),
            'identity_ab_max_abs': float(np.max(np.abs(identity[list(ids)]-baseline[list(ids)]))),
            'zero_ab_max_abs': float(np.max(np.abs(zero[list(ids)]-baseline[list(ids)])))}
