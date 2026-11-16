"""Reproduce fixed checkpoint scores on validation only, using audited source.

No training and no test loader. Read-outs are single-model scores at epochs
selected earlier by validation MA(5), not five-epoch averages.
"""
import argparse,gc,hashlib,json,random,sys,time
from datetime import datetime
from pathlib import Path
from s24_verify_checkpoint_metadata import sha256,numpy_allowlist

def write_report(path,report):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--metadata-report',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    job_raw=args.job.read_bytes();job=json.loads(job_raw.decode('utf-8-sig'))
    assert job['status']=='VALIDATION_INFERENCE_ONLY_TEST_REMAINS_LOCKED'
    assert job['test_access_authorized'] is False and len(job['targets'])==66
    root=Path(job['project_root']).resolve(strict=True)
    out=args.output_dir or root/('s24_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists():raise FileExistsError('Choose a new output folder: '+str(out))
    if not out.resolve().is_relative_to(root):raise ValueError('Output folder must be inside project root')
    out.mkdir(parents=True)
    destination=out/'s24_validation_reproduction.json'
    report={'status':'VALIDATION_ONLY_STARTED','job_sha256':hashlib.sha256(job_raw).hexdigest(),
        'script_sha256':sha256(Path(__file__)),'stage2_report_sha256':job['stage2_report_sha256'],
        'test_images_opened':False,'test_labels_opened':False,'training_performed':False,
        'model_inference_performed':False,'results':[],'preflight_errors':[]}
    started=time.time()
    try:
        metadata_path=args.metadata_report or args.job.with_name('s24_checkpoint_metadata.json')
        if sha256(metadata_path)!=job['stage2_report_sha256']:raise ValueError('Stage-2 report hash mismatch')
        stage2=json.loads(metadata_path.read_text(encoding='utf-8-sig'))
        verified={r['run']:r for r in stage2['results']}
        if len(verified)!=66 or set(verified)!={t['run'] for t in job['targets']}:raise ValueError('Stage-2 endpoint set mismatch')
        for target in job['targets']:
            evidence=verified[target['run']]
            if evidence['status']!='PASS_METADATA' or evidence['issues']:raise ValueError('Stage-2 endpoint did not pass')
            if evidence['checkpoint_sha256']!=target['checkpoint_sha256']:raise ValueError('Stage-2 checkpoint hash mismatch')
            if evidence['metadata']['epoch']!=target['expected_epoch']:raise ValueError('Stage-2 selected epoch mismatch')
        report['stage2_metadata_verified']=True
        if sha256(Path(__file__).with_name('s24_verify_checkpoint_metadata.py'))!=job['metadata_reader_sha256']:
            raise ValueError('Metadata reader differs from verified stage-2 script')
        # Check all audited source/config files before importing project modules.
        for entry in job['source_files']:
            p=Path(entry['path']).resolve(strict=True)
            if not p.is_relative_to(root):raise ValueError('Source path outside project root')
            if sha256(p)!=entry['sha256']:raise ValueError('Source/config changed: '+str(p))
        config_path=Path(job['config_path']).resolve(strict=True)
        if sha256(config_path)!=job['config_sha256']:raise ValueError('Frozen config changed')
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        if str(torch.__version__)!=job['expected_torch_version']:raise RuntimeError('PyTorch version changed since stage 2')
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no automatic CPU fallback')
        device=torch.device(args.device)
        report['runtime']={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
            'cuda':torch.version.cuda,'device':str(device),'cudnn':torch.backends.cudnn.version(),
            'gpu_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None}
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        sys.path.insert(0,str(root/'baseline_platform/src'))
        from surface_crack.config import load_baseline_data_config
        from surface_crack.data import read_pair_records,SurfaceCrackDataset
        from surface_crack.models import UNet
        from surface_crack.metrics import GroupedBinarySegmentationMetrics
        from surface_crack.research_pooling import install_research_pooling
        from surface_crack.spatial import pad_image_and_mask,crop_prediction_to_original
        install_research_pooling()
        config=load_baseline_data_config(config_path)
        # Request val directly: test/external_test sample paths are not resolved.
        records=read_pair_records(config.pairs.path,config.dataset_root,schema=config.schema,
            splits=('val',),expected_pairs_sha256=config.pairs.sha256)
        if any(r.split!='val' for r in records):raise ValueError('Non-validation record requested')
        if len(records)!=job['expected_val_images']:raise ValueError('Validation image count changed')
        if len({r.group_id for r in records})!=job['expected_val_groups']:raise ValueError('Validation group count changed')
        report['validation_data']={'images':len(records),'groups':len({r.group_id for r in records}),
            'dataset_id':config.dataset_id,'pairs_sha256':config.pairs.sha256,
            'dataset_root':str(config.dataset_root),'pairs_path':str(config.pairs.path)}
        report['tolerance']=job['comparison_tolerance']
    except Exception as exc:
        report['status']='PREFLIGHT_FAILED';report['preflight_errors'].append(type(exc).__name__+': '+str(exc))
        write_report(destination,report);print('Preflight failed. Report:',destination);return 2
    for i,target in enumerate(job['targets'],1):
        row={'run':target['run'],'operator':target['operator'],'seed':target['seed'],'budget':target['budget'],
            'checkpoint_epoch':target['expected_epoch'],'checkpoint_sha256':target['checkpoint_sha256'],
            'expected_single_epoch_validation_iou':target['expected_single_epoch_validation_iou']}
        model=None;state=None
        try:
            path=Path(target['checkpoint_path']).resolve(strict=True)
            if not path.is_relative_to(root/'baseline_platform/runs'):raise ValueError('Unexpected checkpoint location')
            if sha256(path)!=target['checkpoint_sha256']:raise ValueError('Checkpoint changed after metadata verification')
            if sha256(Path(target['result_path']))!=target['result_sha256']:raise ValueError('Training result changed')
            original=json.loads(Path(target['result_path']).read_text(encoding='utf-8-sig'))
            if original.get('norm')!=job['required_norm']:raise ValueError('Expected explicit GroupNorm in training result')
            if original.get('amp_dtype')!=job['required_amp_dtype']:raise ValueError('AMP mode changed')
            random.seed(target['seed']);np.random.seed(target['seed']);torch.manual_seed(target['seed'])
            if device.type=='cuda':torch.cuda.manual_seed_all(target['seed'])
            with torch.serialization.safe_globals(numpy_allowlist(np)):
                state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            if state['epoch']!=target['expected_epoch']:raise ValueError('Selected epoch mismatch')
            if state.get('test_or_external_test_used') is not False:raise ValueError('Test-use flag is not false')
            for key,value in target['expected_metadata'].items():
                if state.get(key)!=value:raise ValueError('Checkpoint identity mismatch: '+key)
            # Legacy checkpoints lack a norm field. Use the frozen result profile,
            # checked above, and strict schema/state loading instead of guessing.
            model=UNet(in_channels=3,base_channels=target['expected_metadata']['base_channels'],
                pooling=target['operator'],norm=job['required_norm']).to(device)
            schema={str(k):{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in sorted(model.state_dict().items())}
            schema_hash=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
            if schema_hash!=target['model_schema_sha256']:raise ValueError('Constructed model schema mismatch')
            model.load_state_dict(state['model_state_dict'],strict=True)
            model.eval()
            ordered=list(records);random.Random(target['seed']+1).shuffle(ordered)
            loader=DataLoader(SurfaceCrackDataset(ordered,mask_spec=config.mask_spec),
                batch_size=1,shuffle=False,num_workers=0)
            metrics=GroupedBinarySegmentationMetrics(prediction_space='logits')
            images=0
            with torch.inference_mode():
                for batch in loader:
                    if set(batch['split'])!={'val'}:raise ValueError('Unexpected split in batch')
                    image=batch['image'].to(device,non_blocking=True)
                    mask=batch['mask'].to(device,non_blocking=True)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
                        padded,_,spatial=pad_image_and_mask(image,mask,multiple=16)
                        logits=crop_prediction_to_original(model(padded),spatial)
                    report['model_inference_performed']=True
                    metrics.update(logits.float(),mask,[batch['group_id'][0]])
                    images+=1
            measured=metrics.compute()
            value=float(measured['foreground_iou'])
            gap=value-target['expected_single_epoch_validation_iou']
            if images!=54 or measured['num_groups']!=15:raise ValueError('Incomplete validation evaluation')
            row.update(recomputed_single_epoch_validation_iou=value,difference=gap,
                val_images=images,val_groups=measured['num_groups'],per_group=metrics.per_group(),
                status='PASS_VALIDATION' if abs(gap)<=job['comparison_tolerance'] else 'SCORE_MISMATCH')
        except Exception as exc:
            row.update(status='ERROR',error=type(exc).__name__+': '+str(exc))
        report['results'].append(row)
        report['elapsed_seconds']=time.time()-started
        write_report(destination,report)
        print(f"[{i}/66] {target['run']}: {row['status']}",flush=True)
        model=None;state=None;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    passed=sum(r['status']=='PASS_VALIDATION' for r in report['results'])
    report['summary']={'targets':66,'pass':passed,'fail_or_error':66-passed,
        'maximum_absolute_difference':max((abs(r['difference']) for r in report['results'] if 'difference' in r),default=None)}
    report['status']='VALIDATION_REPRODUCED' if passed==66 else 'VALIDATION_REVIEW_REQUIRED'
    write_report(destination,report)
    print('Report:',destination);print(json.dumps(report['summary'],ensure_ascii=False))
    return 0 if passed==66 else 2

if __name__=='__main__':raise SystemExit(main())
