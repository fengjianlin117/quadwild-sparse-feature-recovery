"""Recount the published tables and verify identities; standard library only."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
MAIN = ['native_B', 'native_B_R', 'self_G_fallback', 'self_G_R_fallback']
METRICS = ['feature_RMS_h', 'surface_RMS_bbox', 'CD_squared_bbox', 'HD_sampled_bbox']
THRESHOLDS = [.01, .025, .05, .1]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_maps():
    result = {}
    with (ROOT/'data/per_model.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            row = dict(row)
            for key in ['readable', 'strict']:
                assert row[key] in ('True', 'False'), (row['id'], key)
                row[key] = row[key] == 'True'
            row['actual_Q'] = int(row['actual_Q']) if row['actual_Q'] else None
            for key in METRICS:
                row[key] = float(row[key]) if row[key] else None
                assert row[key] is None or math.isfinite(row[key]) and row[key] >= 0
            arm = result.setdefault(row['arm'], {})
            assert row['id'] not in arm, ('Duplicate model/config', row['id'], row['arm'])
            assert not row['strict'] or row['readable']
            assert not row['readable'] or all(row[k] is not None for k in METRICS)
            arm[row['id']] = row
    return result


def population(rows):
    rows = list(rows)
    return dict(denominator=len(rows), readable=sum(r['readable'] for r in rows), strict=sum(r['strict'] for r in rows),
        attainment={str(t): sum(r['strict'] and r['feature_RMS_h'] is not None and r['feature_RMS_h'] <= t for r in rows)
                    for t in THRESHOLDS})


def stats(mapping, ids):
    return {key: dict(n=len(ids), median=statistics.median(mapping[i][key] for i in ids),
                      mean=statistics.mean(mapping[i][key] for i in ids)) for key in METRICS}


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15), (a, b)


def verify():
    manifest = read(ROOT/'FILE_MANIFEST.json')
    listed = {r['path'] for r in manifest['files']}
    actual = {str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'FILE_MANIFEST.json' and '.git' not in p.relative_to(ROOT).parts
              and '__pycache__' not in p.relative_to(ROOT).parts}
    assert actual == listed, ('File inventory changed', sorted(actual-listed), sorted(listed-actual))
    for record in manifest['files']:
        path = ROOT/record['path']
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], record['path']
    maps = load_maps(); registry = read(ROOT/'data/model_registry.json')['models']
    eligible = {j['id'] for j in registry if j['source_eligible']}
    assert len(registry) == 999 and len({j['id'] for j in registry}) == 999 and len(eligible) == 980
    summary = read(ROOT/'data/summary.json'); configs = read(ROOT/'data/configurations.json')
    assert set(maps) == set(configs)
    for arm, mapping in maps.items():
        assert set(mapping) == eligible, arm
        assert population(mapping.values()) == summary['populations'][arm], arm
    common = sorted(i for i in eligible if all(maps[a][i]['readable'] for a in MAIN))
    strict = sorted(i for i in common if all(maps[a][i]['strict'] for a in MAIN))
    assert len(common) == 832 and len(strict) == 480
    cohorts = read(ROOT/'data/cohorts.json')
    assert set(common) == set(cohorts['main_common_readable_ids'])
    assert set(strict) == set(cohorts['main_common_strict_ids'])
    for key, ids in [('common_readable_statistics', common), ('common_strict_statistics', strict)]:
        for arm in MAIN:
            actual_stats = stats(maps[arm], ids)
            for metric in METRICS:
                for measure in ['median', 'mean']:
                    close(actual_stats[metric][measure], summary[key][arm][metric][measure])
    rescued = sorted(i for i in eligible if not maps['native_B'][i]['strict'] and maps['self_G_fallback'][i]['strict'])
    assert len(rescued) == 69 and set(rescued) == set(cohorts['newly_strict_ids'])
    assert all(maps['self_G_fallback'][i]['strict'] for i in strict)
    assert population(maps['self_G_fallback'][i] for i in rescued)['attainment']['0.05'] == 10
    assert population(maps['self_G_R_fallback'][i] for i in rescued)['attainment']['0.05'] == 61
    assert population(maps['self_G_R_fallback'][i] for i in strict)['attainment']['0.05'] == 474
    assert 474+61 == summary['populations']['self_G_R_fallback']['attainment']['0.05'] == 535
    assert summary['populations']['self_G_R_fallback']['strict'] == 549
    groups = {group: {j['id'] for j in registry if j.get('native_group') == group}
              for group in ['native_valid', 'geometry_only_failure', 'structural_failure', 'no_readable_output']}
    assert [len(groups[g]) for g in groups] == [480, 269, 83, 148]
    assert set().union(*groups.values()) == eligible
    for group, ids in groups.items():
        value = summary['native_groups'][group]
        assert value['native_n'] == len(ids)
        assert value['D_strict'] == sum(maps['self_G_fallback'][i]['strict'] for i in ids)
    for arm in MAIN:
        for i in eligible:
            if not maps[arm][i]['readable']: assert all(maps[arm][i][k] is None for k in METRICS)
    with (ROOT/'data/main_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [r['arm'] for r in rows] == MAIN
    for row in rows:
        pop = summary['populations'][row['arm']]
        for key in ['denominator', 'readable', 'strict']: assert int(row[key]) == pop[key]
        for t in THRESHOLDS: assert int(row['valid_F_le_'+str(t)]) == pop['attainment'][str(t)]
    for i in eligible:
        for before, after in [('native_B','native_B_R'),('self_G_fallback','self_G_R_fallback')]:
            a,b=maps[before][i],maps[after][i]
            assert a['readable'] == b['readable'] and a['strict'] == b['strict']
            if a['readable']: assert a['actual_Q'] == b['actual_Q']
    reference = read(ROOT/'data/reference_comparisons.json')
    for pair, comparisons in reference['paired_comparisons'].items():
        right,left = pair.split('__vs__')
        for mode,record in comparisons.items():
            paired = [i for i in eligible if maps[left][i]['readable'] and maps[right][i]['readable']
                      and (mode!='strict' or maps[left][i]['strict'] and maps[right][i]['strict'])]
            assert len(paired) == record['n'], (pair,mode)
            for metric,values in record['metrics'].items():
                a=[maps[left][i][metric] for i in paired];b=[maps[right][i][metric] for i in paired]
                for key,value in [('left_median',statistics.median(a)),('right_median',statistics.median(b)),
                                  ('left_mean',statistics.mean(a)),('right_mean',statistics.mean(b))]:
                    close(value,values[key])
    old_fallback = {i:maps['native_B_R'][i] if maps['native_B'][i]['readable'] and not maps['old_MIQ_G'][i]['strict']
                    else maps['old_MIQ_G_R'][i] for i in eligible}
    fair = reference['fair_initialization_comparison']
    shared = [i for i in eligible if old_fallback[i]['readable'] and maps['self_G_R_fallback'][i]['readable']]
    assert len(shared) == fair['common_readable'] == 850
    assert population(old_fallback.values())['strict'] == fair['old_fallback_valid'] == 547
    assert population(old_fallback.values())['attainment']['0.05'] == fair['old_fallback_F005'] == 528
    for metric,values in fair['new_self_fallback_vs_old_MIQ_fallback'].items():
        close(statistics.median(old_fallback[i][metric] for i in shared),values['old_median'])
        close(statistics.median(maps['self_G_R_fallback'][i][metric] for i in shared),values['new_median'])
    assert all(maps['self_G_R'][i]['quad_sha256']==maps['self_G_R_fallback'][i]['quad_sha256']
               for i in eligible if maps['self_G_R'][i]['strict'])
    original_audit = read(ROOT/'provenance/original_run_audit.json')
    assert original_audit['passed'] and original_audit['checked_models'] == 980
    return dict(passed=True, files_verified=len(listed), registered_models=999, eligible_models=980,
                configurations=len(maps), per_model_rows=sum(len(v) for v in maps.values()),
                main_rows=3920, common_readable=832, common_strict=480, full_strict=549, full_F005=535,
                rescued_strict=69, rescued_F005_before_R=10, rescued_F005_after_R=61,
                verification='Recalculation from published records and package hashes; no geometry or backend rerun.')


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
