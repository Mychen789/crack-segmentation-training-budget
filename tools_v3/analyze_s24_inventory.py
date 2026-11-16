"""Audit original-device inventory and prepare metadata-only endpoint verification.

Reads training/validation records only. Does not load model weights or test data.
"""
import csv,hashlib,json,math
from collections import Counter
from pathlib import Path,PureWindowsPath

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/s24'
INVENTORY=OUT/'s24_asset_inventory_20261112_193805.json'
OPS=('max','avg','min','avg_top2','fixed_max_avg','learned_max_avg','t_max_avg','v0_max_mean_min','btsp','diag_partition_matched','stride_conv')
SEEDS=(2026,2027,2028)
raw=INVENTORY.read_bytes()
data=json.loads(raw.decode('utf-8-sig'))
assert data['status']=='INVENTORY_ONLY_NO_INFERENCE' and not data['errors']
assert not any(data[k] for k in ('test_images_opened','test_labels_opened','checkpoint_payloads_loaded','raw_dataset_directories_enumerated'))
runs={r['run']:r for r in data['runs']}
assert len(runs)==len(data['runs'])
assert all(r['metadata'].get('test_or_external_test_used') is False for r in runs.values())

def curve(run):
    history=run['history_primary_validation']
    assert [x['epoch'] for x in history]==list(range(1,len(history)+1)),run['run']
    return [float(x['foreground_iou']) for x in history]

def selection(values,budget):
    assert len(values)>=budget
    means=[sum(values[e-5:e])/5 for e in range(5,budget+1)]
    pos=max(range(len(means)),key=means.__getitem__)
    return pos+5,means[pos]

baseline=[r for r in runs.values() if PureWindowsPath(r['root']).name=='runs']
cosine=[runs[f'cos{b}_unet_gr_deepcrack537_{op}_s{s}'] for b in (30,90) for op in OPS for s in SEEDS]
modern=[runs[f'b90_cnx_gr_deepcrack537_{op}_s{s}'] for op in OPS for s in SEEDS]
formal=baseline+cosine+modern
assert len(baseline)==216 and len(cosine)==66 and len(modern)==33
assert all(r['metadata']['status']=='COMPLETE' for r in formal)
formal_names={r['run'] for r in formal}
checks=[r['run'] for r in runs.values() if r['run'] not in formal_names and '.interrupted_' not in r['run']]
interrupted=[]
for r in runs.values():
    if '.interrupted_' in r['run']:
        name=r['run'].split('.interrupted_')[0]
        replacement=runs[name]
        assert replacement['metadata']['status']=='COMPLETE' and replacement['metadata']['epochs_completed']==90
        interrupted.append({'interrupted':r['run'],'completed_epochs':r['metadata']['epochs_completed'],'complete_replacement':name})

history_checks=[]
for r in formal:
    name=r['run']
    directory='runs' if r in baseline else 'runs-cosine/s21_collect' if r in cosine else 'runs_s22'
    local=ROOT/'data'/directory/(name+'.json')
    if not local.is_file():
        history_checks.append({'run':name,'match':False,'issue':'writing-package result missing'});continue
    expected=json.loads(local.read_text(encoding='utf-8-sig'))
    expected_values=[float(h['validation']['foreground_iou']) for h in expected['history']]
    values=curve(r)
    match=len(values)==len(expected_values) and all(a==b for a,b in zip(values,expected_values))
    history_checks.append({'run':name,'match':match,'epochs':len(values),'source':str(local.relative_to(ROOT))})

targets=[]
truncation_checks=[]
for op in OPS:
    for seed in SEEDS:
        low=runs[f'formal_gn_deepcrack537_unet_{op}_s{seed}']
        high=runs[f'b90_unet_gr_deepcrack537_{op}_s{seed}']
        truncation_checks.append({'operator':op,'seed':seed,'match':curve(low)==curve(high)[:30]})
        for budget,r in ((30,low),(90,high)):
            m=r['metadata'];assert m['status']=='COMPLETE' and m['epochs_completed']==budget
            epoch,value=selection(curve(r),budget)
            assert epoch==r['ma5_selected_epochs'][str(budget)]['epoch']
            assert math.isclose(value,m['best_foreground_iou_ma5'],rel_tol=0,abs_tol=1e-12)
            assert epoch==m['best_epoch_ma5']
            assert m['pooling']==op and m['seed']==seed and m['norm']=='group' and m['warmup_epochs']==5
            assert m['config_sha256']=='00855c79a17db5687ecfe626e75801c45b879ce4199b35af7e2cbb9636e13337'
            best=next(c for c in r['checkpoint_files'] if PureWindowsPath(c['path']).name=='best.pt')
            targets.append({'run':r['run'],'budget':budget,'operator':op,'seed':seed,
                'expected_epoch':epoch,'validation_ma5_value':value,
                'expected_metadata':{k:m[k] for k in ('dataset_id','backbone','pooling','seed','config_sha256','base_channels')},
                'selection_semantics_in_result':m['best_checkpoint_selection'],
                'checkpoint_path':best['path'],'inventory_checkpoint_bytes':best['bytes'],
                'result_path':r['result_file']['path'],'result_sha256':r['result_file']['sha256'],
                'status':'CANDIDATE_INTERNAL_METADATA_NOT_VERIFIED'})

assert all(x['match'] for x in history_checks),[x for x in history_checks if not x['match']]
assert all(x['match'] for x in truncation_checks)
manifest={'status':'METADATA_VERIFICATION_ONLY_NOT_TEST_AUTHORIZATION','inventory_sha256':hashlib.sha256(raw).hexdigest(),
    'project_root':data['project_root'],'targets':targets,
    'source_files':data['code_files']+data['config_files']+data['protocol_artifacts'],
    'test_inference_authorized':False,'checkpoint_metadata_verified':False}
(OUT/'s24_checkpoint_candidates.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
report={'inventory_sha256':manifest['inventory_sha256'],'runs_total':len(runs),'formal_runs':len(formal),
    'formal_baseline':len(baseline),'formal_cosine':len(cosine),'formal_convnext':len(modern),
    'weights_total':sum(len(r['checkpoint_files']) for r in runs.values()),'other_check_runs':checks,
    'interrupted_with_complete_replacements':interrupted,'history_checks':history_checks,
    'truncation_checks':truncation_checks,'target_endpoints':len(targets),
    'target_best_checkpoints':len(targets),'target_selection_semantics':{str(b):dict(Counter(str(t['selection_semantics_in_result']) for t in targets if t['budget']==b)) for b in (30,90)},
    'pending':['checkpoint internal epoch and identity','checkpoint SHA-256','matching trusted code and validation-score verification'],
    'test_data_accessed':False}
(OUT/'s24_inventory_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Formal runs:',len(formal),'= 216 + 66 + 33; total including checks/interrupted:',len(runs))
print('Writing-package histories matched:',len(history_checks),'Truncation matched:',len(truncation_checks))
print('66 candidate checkpoints complete; internal epochs still unverified')
print('Complete replacements:',[x['complete_replacement'] for x in interrupted])
