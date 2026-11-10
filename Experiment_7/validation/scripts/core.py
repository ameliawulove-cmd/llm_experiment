"""Pure numerical and provenance utilities; no inference at import."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verify_basis(basis, dim=1024, rank=16, tolerance=1e-5):
    basis = np.asarray(basis, dtype=float)
    if basis.shape != (dim, rank) or not np.isfinite(basis).all():
        raise ValueError('Invalid basis shape or values')
    error = float(np.max(np.abs(basis.T @ basis - np.eye(rank))))
    if error > tolerance:
        raise ValueError(f'Basis orthogonality error {error}')
    return error


def random_basis(dim, rank, seed):
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, rank)), mode='reduced')[0]


def make_interventions(base, source, basis, seed, tolerance=1e-12):
    base, source, basis = np.asarray(base, float), np.asarray(source, float), np.asarray(basis, float)
    if base.ndim != 1 or source.shape != base.shape:
        raise ValueError('Incompatible base/source activations')
    verify_basis(basis, base.size, basis.shape[1])
    diff = source - base
    das = basis @ (basis.T @ diff)
    random = random_basis(base.size, basis.shape[1], seed)
    natural = random @ (random.T @ diff)
    orthogonal = diff - basis @ (basis.T @ diff)
    norm = np.linalg.norm(das)
    def match(vector):
        length = np.linalg.norm(vector)
        return None if length <= tolerance else vector * (norm / length)
    return {'zero': np.zeros_like(diff), 'identity': np.zeros_like(diff), 'das_natural': das,
            'random_natural': natural, 'random_norm_matched': match(natural),
            'orthogonal_norm_matched': match(orthogonal)}


def metrics(base_logits, intervened_logits, offset, base_target, source_target):
    base, intervened = np.asarray(base_logits, float), np.asarray(intervened_logits, float)
    if base.shape != (2,) or intervened.shape != (2,) or not np.isfinite(np.r_[base, intervened, offset, base_target, source_target]).all():
        raise ValueError('Expected finite A/B logits and targets')
    raw_base, raw_int = float(base[0] - base[1]), float(intervened[0] - intervened[1])
    change, calibrated = raw_int - raw_base, raw_base - offset
    target_difference = source_target - base_target
    return {'raw_margin_base': raw_base, 'raw_margin_intervened': raw_int,
            'raw_margin_change_signed': change, 'raw_margin_change_absolute': abs(change),
            'calibrated_margin_base': calibrated, 'calibrated_margin_intervened': raw_int - offset,
            'calibrated_threshold_crossing': bool(calibrated * (calibrated + change) < 0),
            'restricted_ab_argmax_flip': bool((raw_base > 0) != (raw_int > 0)),
            'directional_agreement': None if target_difference == 0 else bool(np.sign(target_difference) * change > 0),
            'target_difference': target_difference}


def paired_episode_summary(records, seed=8128, replicates=2000):
    groups = defaultdict(dict)
    for row in records:
        pair = (row['base_episode_id'], row['base_trial'], row['source_episode_id'], row['source_trial'])
        if row['condition'] in groups[pair]:
            raise ValueError('Duplicate pair/condition; include random seed in condition name')
        groups[pair][row['condition']] = row
    per_episode = defaultdict(list)
    paired_values = []
    for pair, conditions in groups.items():
        if 'das_natural' not in conditions:
            continue
        controls = [row['raw_margin_change_absolute'] for name, row in conditions.items() if name.startswith('random_natural:')]
        if controls:
            difference = conditions['das_natural']['raw_margin_change_absolute'] - float(np.mean(controls))
            per_episode[pair[0]].append(difference)
            paired_values.append((pair[0], pair[2], difference))
    episodes = {episode: float(np.mean(values)) for episode, values in per_episode.items()}
    values = np.asarray(list(episodes.values()))
    if len(values) < 5:
        return {'episodes': episodes, 'mean': float(values.mean()) if len(values) else None, 'interval': None,
                'warning': 'Fewer than five base episodes; interval suppressed.'}
    all_episodes = sorted({episode for base, source, _ in paired_values for episode in (base, source)})
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        sample = rng.choice(all_episodes, len(all_episodes), replace=True)
        counts = {episode: int(np.count_nonzero(sample == episode)) for episode in all_episodes}
        means, weights = [], []
        for base in episodes:
            selected = [(value, counts[source]) for pair_base, source, value in paired_values if pair_base == base and counts[source] > 0]
            if counts[base] and selected:
                means.append(float(np.average([item[0] for item in selected], weights=[item[1] for item in selected])))
                weights.append(counts[base])
        if weights:
            estimates.append(float(np.average(means, weights=weights)))
    return {'episodes': episodes, 'mean': float(values.mean()), 'interval': np.quantile(estimates, [.025, .975]).tolist(),
            'warning': 'Exploratory dyadic episode-weighted interval; random-basis uncertainty not included. Not confirmatory.'}


def duplicate_report(candidate, historical):
    result = {}
    for field in ('history', 'prompt', 'schedule'):
        known = {object_hash(row[field]) for row in historical if row.get(field) is not None}
        eligible = [row for row in candidate if row.get(field) is not None]
        matches = [row.get('episode_id') for row in eligible if object_hash(row[field]) in known]
        result[field] = {'historical_unique': len(known), 'candidate_count': len(eligible), 'matches': len(matches),
                         'matched_episode_ids': matches, 'status': 'checked' if known else 'not_assessable_no_historical_data'}
    return result
