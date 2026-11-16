"""Freeze the test specification using existing readiness evidence, without opening samples."""
import csv,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts/s24';T=ROOT/'tools_v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
validation=json.loads((A/'s24_validation_reproduction.json').read_text(encoding='utf-8-sig'))
assert validation['status']=='VALIDATION_REPRODUCED' and len(validation['results'])==66
assert all(r['difference']==0 and r['status']=='PASS_VALIDATION' for r in validation['results'])
job=json.loads((A/'s24_validation_job.json').read_text(encoding='utf-8'))
assert validation['job_sha256']==sha(A/'s24_validation_job.json')
cfg=json.loads((A/'source_snapshot/baseline_platform/configs/deepcrack537_train_val.json').read_text(encoding='utf-8'))
csv_path=Path(cfg['artifacts']['pairs_csv']['path'])
assert sha(csv_path)==cfg['artifacts']['pairs_csv']['sha256']
with csv_path.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
ids=[r['sample_id'] for r in rows];gids={r['group_id'] for r in rows};assert len(ids)==107 and len(gids)==30
digest_ids=lambda x:hashlib.sha256(json.dumps(sorted(x),ensure_ascii=False).encode()).hexdigest()
ops=list(dict.fromkeys(t['operator'] for t in job['targets']));assert len(ops)==11
files=[T/'s24_run_frozen_test.py',T/'s24_test_statistics.py',T/'s24_verify_checkpoint_metadata.py',A/'s24_validation_reproduction.json']
plan={**job,'status':'FROZEN_PRETEST_SPECIFICATION','specification_date':'2026-11-13',
    'test_access_authorized':False,'activation':'Only explicit operator launch with --execute-frozen-test creates a separate test-unlock record before sample access.',
    'authorized_scope_on_activation':['DeepCrack537 core test'],'excluded_scope':['external_test','CFD','training','checkpoint or threshold selection'],
    'output_directory':'s24_core_test_20261113','expected_test_images':107,'expected_test_groups':30,
    'test_sample_ids_sha256':digest_ids(ids),'test_group_ids_sha256':digest_ids(gids),
    'pairs_sha256':cfg['artifacts']['pairs_csv']['sha256'],'execution_files':{p.name:sha(p) for p in files},
    'expected_runtime':{'torch':validation['runtime']['torch'],'numpy':validation['runtime']['numpy'],
        'cuda':validation['runtime']['cuda'],'cudnn':validation['runtime']['cudnn'],'gpu':validation['runtime']['gpu_name']},
    'analysis':{'operators':ops,'seeds':[2026,2027,2028],'budgets':[30,90],'delta':.005,
        'criterion_Fcross':.25,'criterion_type':'Predeclared descriptive consistency criterion; not a significance test or replication of test MA(5).',
        'bootstrap_resamples':10000,'bootstrap_seed':424242,
        'bootstrap_scheme':'Resample paired seeds and one shared group-index list across every operator and both budgets. Keep operator set fixed.',
        'matched_single_checkpoint_validation_n_cross':41,'original_window_MA5_n_cross':25,
        'report_regardless_of_criterion':True},
    'selected_endpoint':'Test score of the single checkpoint selected by the right end of the maximal validation-MA(5) window; no selection using test scores.',
    'preparation_test_images_opened':False,'preparation_test_labels_opened':False}
plan.pop('expected_val_images',None);plan.pop('expected_val_groups',None);plan.pop('comparison_tolerance',None)
dest=A/'s24_frozen_test_plan.json'
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
if dest.exists() and dest.read_bytes()!=raw:raise FileExistsError('Frozen specification changed; preserve it and record an amendment instead of overwriting')
dest.write_bytes(raw)
(A/'s24_frozen_test_plan.sha256').write_text(sha(dest)+'  s24_frozen_test_plan.json\n',encoding='utf-8')
package=T/'S24_独立测试执行包.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files+[dest,A/'s24_frozen_test_plan.sha256',T/'S24_独立测试执行说明.md']:z.write(p,p.name)
assert zipfile.ZipFile(package).testzip() is None
print('Frozen pretest specification:',sha(dest));print('Package:',package)
print('No test image or mask was opened. Only frozen CSV split metadata was counted.')
