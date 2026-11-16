"""Independently check stage-2 evidence and compare like-for-like score endpoints."""
import ast,hashlib,itertools,json,statistics
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'artifacts/s24'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((DIR/'s24_checkpoint_candidates.json').read_text(encoding='utf-8'))
report=json.loads((DIR/'s24_checkpoint_metadata.json').read_text(encoding='utf-8-sig'))
inventory=DIR/'s24_asset_inventory_20261112_193805.json'
assert report['manifest_sha256']==sha(DIR/'s24_checkpoint_candidates.json')
assert report['inventory_sha256']==sha(inventory)==manifest['inventory_sha256']
assert report['script_sha256']==sha(ROOT/'tools_v3/s24_verify_checkpoint_metadata.py')
assert report['source_bundle_sha256']==sha(DIR/'s24_matching_source_bundle.zip')
assert report['torch_load_weights_only'] is True
assert all(report[k] is False for k in ('test_images_opened','test_labels_opened','model_inference_performed','validation_inference_verified'))
expected={t['run']:t for t in manifest['targets']}
observed={r['run']:r for r in report['results']}
assert set(expected)==set(observed) and len(observed)==len(report['results'])==66
for name,t in expected.items():
    r=observed[name];m=r['metadata']
    assert r['status']=='PASS_METADATA' and not r['issues']
    assert r['checkpoint_path']==t['checkpoint_path'] and m['epoch']==t['expected_epoch']==r['expected_epoch']
    assert all(m[k]==v for k,v in t['expected_metadata'].items())
    assert abs(r['ma5_value_at_checkpoint']-t['validation_ma5_value'])<1e-12
    assert m['test_or_external_test_used'] is False and len(r['checkpoint_sha256'])==64
assert len({r['checkpoint_sha256'] for r in observed.values()})==66
source_expected={x['path']:x['sha256'] for x in manifest['source_files']}
assert {x['path']:x['sha256'] for x in report['source_files']}==source_expected
snapshot=DIR/'source_snapshot'
for f in report['source_files']:
    p=snapshot/f['bundle_path'];assert sha(p)==f['sha256']
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
ops=list(dict.fromkeys(r['operator'] for r in observed.values()))
for op in ops:
    group=[r for r in observed.values() if r['operator']==op]
    assert len({r['metadata']['model_schema_sha256'] for r in group})==1
    assert len({r['metadata']['model_numel'] for r in group})==1

def endpoint_stats(field):
    means={(b,o):statistics.mean(r[field] for r in observed.values() if r['budget']==b and r['operator']==o) for b in (30,90) for o in ops}
    pairs=[]
    def category(x):return 1 if x>0.005 else -1 if x<-.005 else 0
    for a,b in itertools.combinations(ops,2):
        low=means[30,a]-means[30,b];high=means[90,a]-means[90,b]
        pairs.append({'a':a,'b':b,'delta30':low,'delta90':high,'shift':high-low,'changed':abs(high-low)>.005,
            'sign_flip':low*high<0,'category_change':category(low)!=category(high)})
    return {'field':field,'n_pairs':55,'n_cross':sum(p['changed'] for p in pairs),
        'Fcross':sum(p['changed'] for p in pairs)/55,'Mmax':max(abs(p['shift']) for p in pairs),
        'n_flip':sum(p['sign_flip'] for p in pairs),'n_category_change':sum(p['category_change'] for p in pairs),
        'mean_gain':statistics.mean(means[90,o]-means[30,o] for o in ops),'pairs':pairs}

ma5=endpoint_stats('ma5_value_at_checkpoint')
single=endpoint_stats('single_epoch_validation_iou')
assert ma5['n_cross']==25 and abs(ma5['Mmax']-.018933148520022258)<1e-12
assert single['n_cross']==41
analysis={'status':'ANALYZED_METADATA_AND_STORED_VALIDATION_ONLY','input_metadata_sha256':sha(DIR/'s24_checkpoint_metadata.json'),
    'metadata_targets_verified':66,'source_files_verified':61,'source_python_files_parsed':sum(x['bundle_path'].endswith('.py') for x in report['source_files']),
    'model_schema_stable_within_each_operator':True,'unique_checkpoint_hashes':66,
    'missing_legacy_fields':{str(b):{k:sum(r['metadata'].get(k) is None for r in observed.values() if r['budget']==b) for k in ('norm','warmup_epochs','schedule')} for b in (30,90)},
    'original_ma5_endpoint':ma5,'selected_single_checkpoint_endpoint':single,
    'max_absolute_single_minus_ma5':max(abs(r['single_epoch_validation_iou']-r['ma5_value_at_checkpoint']) for r in observed.values()),
    'validation_inference_performed_locally':False,'test_inference_performed':False,
    'next_step':'Run fixed 66-checkpoint validation inference against the stored single-epoch scores, not against MA(5).'}
(DIR/'s24_checkpoint_analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
job={'status':'VALIDATION_INFERENCE_ONLY_TEST_REMAINS_LOCKED','project_root':manifest['project_root'],
    'stage2_report_sha256':sha(DIR/'s24_checkpoint_metadata.json'),'source_files':report['source_files'],
    'metadata_reader_sha256':report['script_sha256'],'expected_torch_version':report['torch_version'],
    'config_path':manifest['project_root']+'\\baseline_platform\\configs\\deepcrack537_train_val.json',
    'config_sha256':next(iter(expected.values()))['expected_metadata']['config_sha256'],
    'required_norm':'group','required_amp_dtype':'bfloat16','expected_val_images':54,'expected_val_groups':15,
    'comparison_tolerance':1e-8,'test_access_authorized':False,
    'targets':[{**t,'status':'METADATA_VERIFIED_PENDING_VALIDATION','checkpoint_sha256':observed[t['run']]['checkpoint_sha256'],
        'expected_single_epoch_validation_iou':observed[t['run']]['single_epoch_validation_iou'],
        'model_schema_sha256':observed[t['run']]['metadata']['model_schema_sha256']} for t in manifest['targets']]}
(DIR/'s24_validation_job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
print('All 66 endpoint identities and 61 source hashes independently verified.')
print('MA(5):',ma5['n_cross'],'/55, Mmax',ma5['Mmax'])
print('Selected single-epoch validation:',single['n_cross'],'/55, Mmax',single['Mmax'])
print('Validation-only job prepared; no model inference or test access occurred.')
