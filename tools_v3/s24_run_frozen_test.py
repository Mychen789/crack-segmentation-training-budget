"""Explicitly launch or resume the fixed 66-model core-test evaluation.

The --execute-frozen-test flag acknowledges first use of the held-out test data.
No training, checkpoint selection, threshold tuning, or external-test access.
"""
import argparse,gc,hashlib,json,random,secrets,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
from s24_verify_checkpoint_metadata import sha256,numpy_allowlist

def now():return datetime.now(timezone.utc).isoformat()
def save(path,data):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def digest_ids(ids):return hashlib.sha256(json.dumps(sorted(ids),ensure_ascii=False).encode()).hexdigest()

def pack_results(out,plan_path):
    bundle=out/'S24_测试结果回传.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in (out/'s24_test_confirmation.json',out/'s24_test_execution_freeze.json',plan_path):z.write(p,p.name)
    print('Return bundle:',bundle)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--execute-frozen-test',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if not args.execute_frozen_test:
        print('No test access. Read the plan, then supply --execute-frozen-test to acknowledge this one-time evaluation.');return 0
    raw=args.plan.read_bytes();plan=json.loads(raw.decode('utf-8-sig'));ph=hashlib.sha256(raw).hexdigest()
    if plan['status']!='FROZEN_PRETEST_SPECIFICATION':raise ValueError('Not a frozen specification')
    for name,h in plan['execution_files'].items():
        if sha256(args.plan.parent/name)!=h:raise ValueError('Execution file changed: '+name)
    root=Path(plan['project_root']).resolve(strict=True);out=root/plan['output_directory']
    if not out.resolve().is_relative_to(root):raise ValueError('Output outside project root')
    dest=out/'s24_test_confirmation.json';freeze=out/'s24_test_execution_freeze.json'
    if out.exists():
        if not args.resume:raise FileExistsError('Output exists. Use --resume for the same frozen batch; do not create another test batch.')
        report=json.loads(dest.read_text(encoding='utf-8'))
        if report['plan_sha256']!=ph:raise ValueError('Resume plan changed')
        if freeze.exists():
            frozen=json.loads(freeze.read_text(encoding='utf-8'))
            if frozen['plan_sha256']!=ph or frozen['execution_files']!=plan['execution_files']:raise ValueError('Resume execution changed')
        elif report['results'] or report.get('test_access_started_utc'):raise ValueError('Missing execution freeze after access began')
        if report['status']=='COMPLETE':
            pack_results(out,args.plan)
            print('This frozen test batch already completed. Existing report:',dest);return 0
    else:
        out.mkdir()
        report={'status':'PRECHECK','plan_sha256':ph,'started_utc':now(),'training_performed':False,
            'test_images_opened':False,'test_labels_opened':False,'external_test_accessed':False,'results':[],'errors':[]}
        save(dest,report)
    try:
        validation=json.loads((args.plan.parent/'s24_validation_reproduction.json').read_text(encoding='utf-8-sig'))
        if validation['status']!='VALIDATION_REPRODUCED' or len(validation['results'])!=66:raise ValueError('Validation not complete')
        if any(r['status']!='PASS_VALIDATION' or r['difference']!=0 for r in validation['results']):raise ValueError('Validation failed')
        validated={r['run']:r for r in validation['results']}
        if len({t['run'] for t in plan['targets']})!=66 or set(validated)!={t['run'] for t in plan['targets']}:raise ValueError('Target matrix differs from validation')
        for t in plan['targets']:
            if validated[t['run']]['checkpoint_sha256']!=t['checkpoint_sha256'] or validated[t['run']]['checkpoint_epoch']!=t['expected_epoch']:raise ValueError('Target is not the validated checkpoint')
        for entry in plan['source_files']:
            p=Path(entry['path']).resolve(strict=True)
            if not p.is_relative_to(root) or sha256(p)!=entry['sha256']:raise ValueError('Audited source changed: '+str(p))
        config_path=Path(plan['config_path']).resolve(strict=True)
        if sha256(config_path)!=plan['config_sha256']:raise ValueError('Config changed')
        # Verify the entire matrix before resolving any test sample paths.
        for t in plan['targets']:
            p=Path(t['checkpoint_path']).resolve(strict=True)
            if not p.is_relative_to(root/'baseline_platform/runs') or sha256(p)!=t['checkpoint_sha256']:raise ValueError('Checkpoint changed: '+t['run'])
            if sha256(Path(t['result_path']))!=t['result_sha256']:raise ValueError('Training result changed')
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        if str(torch.__version__)!=plan['expected_torch_version'] or not torch.cuda.is_available():raise RuntimeError('Require verified PyTorch and CUDA')
        runtime={'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
            'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0)}
        if any(runtime[k]!=v for k,v in plan['expected_runtime'].items()):raise ValueError('Runtime differs from verified validation environment')
        if args.resume and report.get('runtime') and runtime!=report['runtime']:raise ValueError('Runtime changed during test batch')
        report['runtime']=runtime;device=torch.device('cuda')
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        sys.path.insert(0,str(root/'baseline_platform/src'))
        from surface_crack.config import load_baseline_data_config
        from surface_crack.data import read_pair_records,SurfaceCrackDataset
        from surface_crack.split_guard import SplitGuard
        from surface_crack.models import UNet
        from surface_crack.research_pooling import install_research_pooling
        from surface_crack.metrics import GroupedBinarySegmentationMetrics
        from surface_crack.spatial import pad_image_and_mask,crop_prediction_to_original
        install_research_pooling();config=load_baseline_data_config(config_path)
        if config.pairs.sha256!=plan['pairs_sha256']:raise ValueError('Pairs CSV identity changed')
        # Record the separate, explicit core-test unlock before requesting test rows.
        token_path=out/'core_test_token.txt'
        if freeze.exists():
            frozen=json.loads(freeze.read_text(encoding='utf-8'));token=token_path.read_text(encoding='utf-8')
            if frozen['plan_sha256']!=ph or hashlib.sha256(token.encode()).hexdigest()!=frozen['unlock_token_sha256']:raise ValueError('Freeze/token mismatch')
        else:
            token=secrets.token_hex(32);token_path.write_text(token,encoding='utf-8')
            frozen={'status':'CORE_TEST_EXECUTION_FROZEN','created_utc':now(),'plan_sha256':ph,
                'execution_files':plan['execution_files'],'validation_report_sha256':plan['execution_files']['s24_validation_reproduction.json'],
                'authorized_splits':['test'],'acknowledgment':'Operator supplied --execute-frozen-test',
                'unlock_token_sha256':hashlib.sha256(token.encode()).hexdigest(),'scope':'Fixed 66 selected checkpoints, DeepCrack537 core test only'}
            save(freeze,frozen)
        guard=SplitGuard(unlock_token_sha256=frozen['unlock_token_sha256'])
        report['test_access_started_utc']=report.get('test_access_started_utc',now());save(dest,report)
        records=read_pair_records(config.pairs.path,config.dataset_root,schema=config.schema,splits=('test',),
            split_guard=guard,unlock_token=token,expected_pairs_sha256=config.pairs.sha256)
        if len(records)!=107 or len({r.group_id for r in records})!=30 or any(r.split!='test' for r in records):raise ValueError('Unexpected test matrix')
        if digest_ids([r.sample_id for r in records])!=plan['test_sample_ids_sha256'] or digest_ids({r.group_id for r in records})!=plan['test_group_ids_sha256']:raise ValueError('Test identities changed')
        report['test_data']={'images':107,'groups':30,'dataset_id':config.dataset_id,'pairs_sha256':config.pairs.sha256}
    except Exception as exc:
        report['status']='PRECHECK_FAILED';report['errors'].append({'time':now(),'error':type(exc).__name__+': '+str(exc)})
        save(dest,report);print('Precheck failed; no automatic repair. Report:',dest);return 2
    # Technical resume reuses completed rows; it never chooses different models.
    completed={r['run'] for r in report['results'] if r['status']=='COMPLETE'}
    report['results']=[r for r in report['results'] if r['status']=='COMPLETE']
    report['status']='EVALUATING';save(dest,report);started=time.time()
    for i,t in enumerate(plan['targets'],1):
        if t['run'] in completed:continue
        model=None;state=None
        try:
            with torch.serialization.safe_globals(numpy_allowlist(np)):
                state=torch.load(t['checkpoint_path'],map_location='cpu',weights_only=True,mmap=True)
            if state['epoch']!=t['expected_epoch'] or state.get('test_or_external_test_used') is not False:raise ValueError('Checkpoint epoch/history flag mismatch')
            if any(state.get(k)!=v for k,v in t['expected_metadata'].items()):raise ValueError('Checkpoint identity mismatch')
            random.seed(t['seed']);np.random.seed(t['seed']);torch.manual_seed(t['seed']);torch.cuda.manual_seed_all(t['seed'])
            model=UNet(in_channels=3,base_channels=t['expected_metadata']['base_channels'],pooling=t['operator'],norm='group').to(device)
            schema={str(k):{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in sorted(model.state_dict().items())}
            if hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()!=t['model_schema_sha256']:raise ValueError('Constructed model schema mismatch')
            model.load_state_dict(state['model_state_dict'],strict=True);model.eval()
            ordered=list(records);random.Random(t['seed']+1).shuffle(ordered)
            loader=DataLoader(SurfaceCrackDataset(ordered,mask_spec=config.mask_spec,split_guard=guard,unlock_token=token),batch_size=1,shuffle=False,num_workers=0)
            metrics=GroupedBinarySegmentationMetrics(prediction_space='logits');images=0
            report['test_sample_access_attempted']=True;save(dest,report)
            with torch.inference_mode():
                for batch in loader:
                    if not report['test_images_opened']:
                        report['test_images_opened']=True;report['test_labels_opened']=True;save(dest,report)
                    if set(batch['split'])!={'test'}:raise ValueError('Unexpected split')
                    image=batch['image'].to(device,non_blocking=True);mask=batch['mask'].to(device,non_blocking=True)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
                        padded,_,spatial=pad_image_and_mask(image,mask,multiple=16)
                        logits=crop_prediction_to_original(model(padded),spatial)
                    metrics.update(logits.float(),mask,[batch['group_id'][0]]);images+=1
            scores=metrics.compute()
            if images!=107 or scores['num_groups']!=30:raise ValueError('Incomplete test evaluation')
            report['results'].append({'run':t['run'],'budget':t['budget'],'operator':t['operator'],'seed':t['seed'],
                'checkpoint_epoch':t['expected_epoch'],'checkpoint_sha256':t['checkpoint_sha256'],'status':'COMPLETE',
                'foreground_iou':float(scores['foreground_iou']),'per_group':metrics.per_group()})
            save(dest,report);print(f"[{i}/66] {t['run']}: COMPLETE",flush=True)
        except Exception as exc:
            report['status']='TECHNICAL_INTERRUPTION';report['errors'].append({'run':t['run'],'time':now(),'error':type(exc).__name__+': '+str(exc)})
            save(dest,report);print('Stopped on technical error. Resume only with the same plan/code/weights. Report:',dest);return 2
        finally:
            model=None;state=None;gc.collect();torch.cuda.empty_cache()
    from s24_test_statistics import analyze
    report['statistics']=analyze(report['results'],plan['analysis'])
    report['status']='COMPLETE';report['finished_utc']=now();report['last_session_seconds']=time.time()-started
    save(dest,report)
    pack_results(out,args.plan)
    print('All66 completed. Descriptive criterion met:',report['statistics']['criterion']['met'])
    return 0

if __name__=='__main__':raise SystemExit(main())
