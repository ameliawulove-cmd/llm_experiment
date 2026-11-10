"""Manifest-locked FVD collection, norm-matched interventions and paired analysis."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from core import file_hash, metrics, object_hash, random_basis, verify_basis
from model_adapter import completion_id, prepared_prompt, prompt_hash

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
HIST_ENV = Path.home() / 'Desktop/llm experiment/exp5_0.6b/results/sequential/environments/environments.json'


def schedules(entries):
    result = []
    for entry in entries:
        rng = np.random.default_rng(entry['schedule_seed'])
        high = 'A' if rng.random() < .5 else 'B'
        reversal = 15 + int(rng.integers(-3, 4))
        uniforms = rng.random((30, 2))
        schedule = []
        for index in range(30):
            current = high if index < reversal else ('B' if high == 'A' else 'A')
            pa, pb = (.8, .2) if current == 'A' else (.2, .8)
            schedule.append({'trial': index + 1, 'p_A': pa, 'p_B': pb,
                             'potential_A': int(uniforms[index, 0] < pa),
                             'potential_B': int(uniforms[index, 1] < pb)})
        result.append({'episode_id': entry['episode_id'], 'reversal_trial': reversal + 1,
                       'initial_high_arm': high, 'schedule': schedule})
    return result


def pairs(entries, development=False):
    if development:
        return [(entries[0]['episode_id'], 1, entries[1]['episode_id'], 1, 0)]
    chosen = sorted(row['episode_id'] for row in entries)
    if len(chosen) != 20 or len(set(chosen)) != 20:
        raise ValueError('Expected 20 unique final episode IDs')
    return [(chosen[group*2+side], trial, chosen[group*2+1-side], trial, group)
            for group in range(10) for side in range(2) for trial in range(6, 16)]


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (10,):
        raise ValueError('Expected ten independent episode-pair clusters')
    rng = np.random.default_rng(8128)
    samples = np.mean(rng.choice(values, (10000, 10), replace=True), axis=1)
    return {'mean': float(values.mean()), 'interval_95_percentile': np.quantile(samples, [.025, .975]).tolist(),
            'clusters': values.tolist(), 'seed': 8128, 'replicates': 10000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'final'], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = ROOT / 'results' / ('preflight_run_v2' if args.mode == 'preflight' else 'confirmatory_run')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    lock_path = ROOT / 'results/artifact_lock.json'
    lock = json.loads(lock_path.read_text())
    for path, expected in lock['files'].items():
        if file_hash(path) != expected:
            raise ValueError(f'Artifact hash mismatch: {path}')
    manifest = json.loads((ROOT / 'results/episode_manifest.json').read_text())
    entries = manifest['development'][:2] if args.mode == 'preflight' else manifest['confirmatory']
    historical = [json.loads(line) for line in (PROJECT / 'data/exp6_full/trials.jsonl').read_text().splitlines() if line]
    environment = json.loads(HIST_ENV.read_text())
    by_id = {ep['episode_id']: ep for ep in environment['episodes']}
    if any(row['episode_id'] not in by_id or
           row['reward'] != by_id[row['episode_id']]['schedule'][row['trial']-1]['potential_'+row['chosen_arm']] or
           row['reversal_trial'] != by_id[row['episode_id']]['reversal_trial'] for row in historical):
        raise ValueError('Historical environment does not match observed trials')
    episodes = schedules(entries)
    known_ids = {row['episode_id'] for row in historical}
    known_schedules = {object_hash(ep['schedule']) for ep in environment['episodes']}
    new_hashes = [object_hash(ep['schedule']) for ep in episodes]
    if known_ids & {ep['episode_id'] for ep in episodes} or len(set(new_hashes)) != len(new_hashes) or set(new_hashes) & known_schedules:
        raise ValueError('Episode or schedule overlap')
    if args.mode == 'final':
        preflight = ROOT / 'results/preflight_run_v2/summary.json'
        if not preflight.exists() or json.loads(preflight.read_text()).get('status') != 'PASS':
            raise RuntimeError('Development preflight has not passed')
        if not (ROOT / 'FINAL_PROTOCOL.md').is_file():
            raise RuntimeError('Final protocol missing')
        frozen = json.loads((ROOT / 'REPRODUCIBILITY_MANIFEST.json').read_text())
        for path, expected in frozen['files'].items():
            if file_hash(path) != expected:
                raise ValueError(f'Frozen protocol or implementation changed: {path}')
        prior = json.loads(preflight.read_text())
        if prior['script_sha256'] != file_hash(Path(__file__)) or prior['protocol_sha256'] != file_hash(ROOT / 'FINAL_PROTOCOL.md'):
            raise ValueError('Preflight did not test current frozen implementation')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path.home() / 'Desktop/llm experiment/exp1-4'))
    sys.path.insert(0, str(PROJECT / 'data/causal_rpe_bandit_code'))
    from exp6_fvd_bandit import fvd_declare_a_raw, fvd_declare_b_raw, fvd_choice_raw
    from sequential_bandit_rpe import BIN_LABELS, BINS, softmax
    snapshot = next(Path(path).parent for path in lock['files'] if path.endswith('/model.safetensors'))
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float32).to('mps').eval()
    if len(model.model.layers) != 28 or model.config.hidden_size != 1024 or str(model.config._attn_implementation) != 'sdpa':
        raise RuntimeError('Model configuration differs from protocol')
    basis = np.load(PROJECT / 'results/exp7h/R_choice_tendency_L27_k16.npy', allow_pickle=False).astype(float)
    verify_basis(basis)
    random_bases = {seed: random_basis(1024, 16, seed) for seed in range(830000, 830010)}
    output.mkdir()
    count = 0
    def run(raw, labels, need_state=False, replacement=None):
        nonlocal count
        prompt = prepared_prompt(tokenizer, raw)
        ids = [completion_id(tokenizer, prompt, label) for label in labels]
        enc = {key: val.to('mps') for key, val in tokenizer(prompt, return_tensors='pt', add_special_tokens=False).items()}
        observed, handles = {'replace_count': 0, 'next_count': 0}, []
        if replacement is not None:
            def patch(_module, _args, value):
                observed['replace_count'] += 1
                tensor = value[0] if isinstance(value, tuple) else value
                modified = tensor.clone()
                modified[0, -1, :] = torch.as_tensor(replacement, device=tensor.device, dtype=tensor.dtype)
                return (modified,) + value[1:] if isinstance(value, tuple) else modified
            def next_input(_module, inputs):
                observed['next_count'] += 1
                observed['next'] = inputs[0][0, -1].detach().float().cpu().numpy()
            handles = [model.model.layers[26].register_forward_hook(patch),
                       model.model.layers[27].register_forward_pre_hook(next_input)]
        try:
            with torch.inference_mode():
                out = model(**enc, output_hidden_states=need_state, return_dict=True)
            count += 1
        finally:
            for handle in handles:
                handle.remove()
        logits = out.logits[0, -1].detach().float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError('Non-finite logits')
        state = out.hidden_states[27][0, -1].detach().float().cpu().numpy() if need_state else None
        if replacement is not None:
            if observed['replace_count'] != 1 or observed['next_count'] != 1:
                raise ValueError('Hook count mismatch')
            observed['next_error'] = float(np.max(np.abs(observed['next'] - replacement)))
            if observed['next_error'] > .01:
                raise ValueError('Next-block input does not match intended activation')
        return logits[ids], state, prompt, ids, observed
    def ab_logits(raw):
        return run(raw, ('A', 'B'))[0]
    trials, activation = {}, {}
    row_count = 2 if args.mode == 'preflight' else 30
    for entry, episode in zip(entries, episodes):
        rng = np.random.default_rng(entry['behavior_seed'])
        bins = rng.permutation(BINS)
        neutral = BIN_LABELS[len(BIN_LABELS)//2]
        base_a = run(fvd_declare_a_raw([], bins), BIN_LABELS)[0]
        base_b = run(fvd_declare_b_raw([], bins, neutral), BIN_LABELS)[0]
        base_choice = ab_logits(fvd_choice_raw([], bins, neutral, neutral))
        offset = float(base_choice[0] - base_choice[1])
        history = []
        for item in episode['schedule'][:row_count]:
            prior_history = list(history)
            def label(raw, baseline):
                values = run(raw, BIN_LABELS)[0]
                estimate = float(softmax(values - baseline) @ bins)
                return BIN_LABELS[int(np.argmin(np.abs(bins-estimate)))]
            la = label(fvd_declare_a_raw(history, bins), base_a)
            lb = label(fvd_declare_b_raw(history, bins, la), base_b)
            raw = fvd_choice_raw(history, bins, la, lb)
            choice, state, prompt, token_ids, _ = run(raw, ('A', 'B'), need_state=True)
            calibrated = choice - base_choice
            exploratory = bool(rng.random() < .1)
            selected = ('A','B')[int(rng.integers(2))] if exploratory else ('A','B')[int(np.argmax(calibrated))]
            reward = item['potential_'+selected]
            key = (episode['episode_id'], item['trial'])
            trials[key] = {'episode_id': key[0], 'trial': key[1], 'history': prior_history,
                           'prompt': prompt, 'prompt_hash': prompt_hash(prompt), 'raw_prompt': raw,
                           'token_ids': token_ids, 'baseline_logits': choice.tolist(), 'calibration_offset': offset,
                           'choice_tendency': float(calibrated[0]-calibrated[1]), 'chosen_arm': selected,
                           'reward': reward, 'reversal_trial': episode['reversal_trial'],
                           'bin_values': bins.tolist(), 'declared_labels': [la, lb]}
            activation[key] = state.astype(np.float64)
            history.append((selected, reward))
    all_pairs = pairs(entries, args.mode == 'preflight')
    if args.mode == 'preflight':
        first = next(iter(trials))
        original = trials[first]
        identity, _, _, ids, hook = run(original['raw_prompt'], ('A', 'B'), replacement=activation[first])
        if ids != original['token_ids'] or not np.allclose(identity, original['baseline_logits'], rtol=0, atol=.01) or hook['next_error'] > .01:
            raise ValueError('Development identity preflight failed')
    records = []
    for base_ep, base_trial, source_ep, source_trial, cluster in all_pairs:
        base_key, source_key = (base_ep, base_trial), (source_ep, source_trial)
        base, source = trials[base_key], trials[source_key]
        base_state, source_state = activation[base_key], activation[source_key]
        diff = source_state - base_state
        das_delta = basis @ (basis.T @ diff)
        norm = np.linalg.norm(das_delta)
        interventions = [('das', None, das_delta)]
        for seed, random in random_bases.items():
            projected = random @ (random.T @ diff)
            if np.linalg.norm(projected) < 1e-12:
                raise ValueError('Degenerate random projection')
            interventions.append(('random', seed, projected * norm / np.linalg.norm(projected)))
        for condition, seed, delta in interventions:
            expected = base_state + delta
            out, _, _, ids, observed = run(base['raw_prompt'], ('A','B'), replacement=expected)
            if ids != base['token_ids']:
                raise ValueError('Completion token mismatch')
            actual_norm = float(np.linalg.norm((observed.pop('next').astype(float)-base_state)))
            if not np.isclose(actual_norm, norm, atol=1e-6, rtol=1e-4):
                raise ValueError('Intervention norms not matched')
            record = {'cluster': cluster, 'condition': condition, 'random_seed': seed,
                      'base_episode_id': base_ep, 'base_trial': base_trial,
                      'source_episode_id': source_ep, 'source_trial': source_trial,
                      'base_prompt_hash': base['prompt_hash'], 'source_prompt_hash': source['prompt_hash'],
                      'base_activation_ref': f'{base_ep}:{base_trial}', 'source_activation_ref': f'{source_ep}:{source_trial}',
                      'baseline_logits': base['baseline_logits'], 'intervened_logits': out.tolist(),
                      'calibration_offset': base['calibration_offset'], 'base_target': base['choice_tendency'],
                      'source_target': source['choice_tendency'], 'intended_intervention_norm': float(norm),
                      'applied_intervention_norm': actual_norm, 'next_block_max_abs_error': observed['next_error'],
                      'hook_counts': [observed['replace_count'], observed['next_count']],
                      'artifact_lock_sha256': file_hash(lock_path)}
            record.update(metrics(record['baseline_logits'], record['intervened_logits'], record['calibration_offset'], record['base_target'], record['source_target']))
            records.append(record)
    expected_n = len(all_pairs)*11
    if len(records) != expected_n or any(row['hook_counts'] != [1,1] for row in records):
        raise ValueError('Incomplete pair-condition grid')
    hist_history = {object_hash(row['history']) for row in historical}
    hist_prompts = {object_hash(fvd_choice_raw(row['history'], np.asarray(row['bin_values_by_label'], float),
                                               row['q_stated_A_label'], row['q_stated_B_label'])) for row in historical}
    history_duplicates = sum(object_hash(row['history']) in hist_history for row in trials.values())
    prompt_duplicates = sum(object_hash(row['raw_prompt']) in hist_prompts for row in trials.values())
    empty_histories = sum(len(row['history']) == 0 for row in trials.values())
    result = {'status': 'PASS', 'mode': args.mode, 'forward_passes': count,
              'wall_seconds': time.monotonic()-started, 'pair_count': len(all_pairs), 'record_count': len(records),
              'history_exact_overlap_count': history_duplicates, 'prompt_exact_overlap_count': prompt_duplicates,
              'empty_history_count': empty_histories, 'trial_count': len(trials),
              'historical_environment': str(HIST_ENV), 'historical_environment_sha256': file_hash(HIST_ENV),
              'protocol_sha256': file_hash(ROOT/'FINAL_PROTOCOL.md'), 'manifest_sha256': file_hash(ROOT/'results/episode_manifest.json'),
              'script_sha256': file_hash(Path(__file__)), 'artifact_lock_sha256': file_hash(lock_path),
              'torch': str(torch.__version__), 'transformers': transformers.__version__, 'model_revision': lock['snapshot_revision'],
              'attention': str(model.config._attn_implementation), 'dtype': str(next(model.parameters()).dtype),
              'device': str(next(model.parameters()).device), 'chat_template_hash': prompt_hash(tokenizer.chat_template),
              'note': 'Historical potential rewards provenance inferred from exact observed reward/reversal match; not established uniquely.'}
    if args.mode == 'final':
        cluster_values = []
        for cluster in range(10):
            group = [row for row in records if row['cluster'] == cluster]
            deltas = []
            for pair in [pair for pair in all_pairs if pair[4] == cluster]:
                matching = [row for row in group if (row['base_episode_id'], row['base_trial'], row['source_episode_id'], row['source_trial']) == pair[:4]]
                das = next(row for row in matching if row['condition'] == 'das')
                controls = [row for row in matching if row['condition'] == 'random']
                if len(controls) != 10:
                    raise ValueError('Incomplete controls')
                deltas.append(das['raw_margin_change_absolute'] - np.mean([row['raw_margin_change_absolute'] for row in controls]))
            cluster_values.append(np.mean(deltas))
        result['primary_cluster_bootstrap'] = bootstrap(cluster_values)
    with (output/'episodes.json').open('x') as handle:
        json.dump(episodes, handle, indent=2)
    with (output/'trials.jsonl').open('x') as handle:
        for row in trials.values(): handle.write(json.dumps(row, allow_nan=False)+'\n')
    with (output/'records.jsonl').open('x') as handle:
        for row in records: handle.write(json.dumps(row, allow_nan=False)+'\n')
    with (output/'activations.npz').open('xb') as handle:
        np.savez_compressed(handle, **{f'{key[0]}:{key[1]}': value for key, value in activation.items()})
    for name in ('episodes.json','trials.jsonl','records.jsonl','activations.npz'):
        result[name+'_sha256'] = file_hash(output/name)
    with (output/'summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
