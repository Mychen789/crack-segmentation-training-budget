"""Original-device stage 2: verify the 66 candidate checkpoint identities on CPU.

No model construction, training, data loader, image access, or test inference.
Only trusted training checkpoint files named by the supplied manifest are read.
"""
import argparse,datetime,gc,hashlib,importlib,json,os,sys,zipfile
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def within(path,root):
    path=path.resolve(strict=True)
    if not path.is_relative_to(root):raise ValueError('Path outside declared project: '+str(path))
    return path

def numpy_allowlist(np):
    reconstruct=importlib.import_module('numpy.core.multiarray')._reconstruct
    scalar=importlib.import_module('numpy.core.multiarray').scalar
    return [np.ndarray,np.dtype,type(np.dtype('uint32')),type(np.dtype('float64')),
        (reconstruct,'numpy.core.multiarray._reconstruct'),
        (reconstruct,'numpy._core.multiarray._reconstruct'),
        (scalar,'numpy.core.multiarray.scalar'),(scalar,'numpy._core.multiarray.scalar')]

def inspect_checkpoint(path,torch,np):
    # mmap avoids eagerly copying optimizer/model tensor storage. weights_only is
    # never disabled on failure. Only NumPy array/RNG primitive types are allowed.
    with torch.serialization.safe_globals(numpy_allowlist(np)):
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if not isinstance(payload,dict):raise TypeError('Checkpoint is not a dictionary')
    keys=('epoch','dataset_id','backbone','pooling','seed','config_sha256','base_channels',
          'warmup_epochs','norm','amp_dtype','schedule','best_epoch_ma','best_fg_iou_ma',
          'test_or_external_test_used')
    meta={k:payload.get(k) for k in keys}
    state=payload.get('model_state_dict')
    if not isinstance(state,dict) or not state:raise TypeError('model_state_dict missing/empty')
    schema={str(k):{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in sorted(state.items())}
    meta['model_schema_sha256']=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
    meta['model_tensor_count']=len(state)
    meta['model_numel']=sum(v.numel() for v in state.values())
    meta['top_level_keys']=list(payload)
    history=payload.get('history',[])
    curve=[{'epoch':int(x['epoch']),'foreground_iou':float(x['validation']['foreground_iou'])} for x in history]
    del payload,state
    return meta,curve

def assess(meta,history,target,result):
    issues=[]
    if meta.get('epoch')!=target['expected_epoch']:issues.append('CHECKPOINT_EPOCH_DOES_NOT_MATCH_SELECTION')
    for key,value in target['expected_metadata'].items():
        if meta.get(key)!=value:issues.append('IDENTITY_MISMATCH:'+key)
    if meta.get('test_or_external_test_used') is not False:issues.append('TEST_USE_FLAG_NOT_EXPLICITLY_FALSE')
    expected=[{'epoch':int(x['epoch']),'foreground_iou':float(x['validation']['foreground_iou'])} for x in result.get('history',[])][:target['expected_epoch']]
    if history!=expected:issues.append('CHECKPOINT_VALIDATION_HISTORY_PREFIX_MISMATCH')
    value=None
    if len(history)>=5:
        value=sum(x['foreground_iou'] for x in history[-5:])/5
        if abs(value-target['validation_ma5_value'])>1e-12:issues.append('MA5_VALUE_AT_CHECKPOINT_MISMATCH')
    else:issues.append('CHECKPOINT_HISTORY_TOO_SHORT')
    return issues,value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    manifest_raw=args.manifest.read_bytes()
    manifest=json.loads(manifest_raw.decode('utf-8-sig'))
    assert manifest['status']=='METADATA_VERIFICATION_ONLY_NOT_TEST_AUTHORIZATION'
    assert manifest['test_inference_authorized'] is False
    assert len(manifest['targets'])==66
    root=Path(manifest['project_root']).resolve(strict=True)
    out=args.output_dir or root/('s24_checkpoint_metadata_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists():raise FileExistsError('Choose a new output folder: '+str(out))
    if not out.resolve().is_relative_to(root):raise ValueError('Output folder must be within the project root')
    # This environment change is process-local and makes CPU-only intent explicit.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import numpy as np
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    report={'status':'CHECKPOINT_METADATA_ONLY_NO_INFERENCE','manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
        'inventory_sha256':manifest['inventory_sha256'],'script_sha256':sha256(Path(__file__)),
        'python':sys.version,'torch_version':torch.__version__,'numpy_version':np.__version__,
        'test_images_opened':False,'test_labels_opened':False,'model_inference_performed':False,
        'checkpoint_payloads_loaded':True,'torch_load_weights_only':True,'results':[],
        'source_files':[],'source_errors':[]}
    for i,target in enumerate(manifest['targets'],1):
        row={'run':target['run'],'budget':target['budget'],'operator':target['operator'],'seed':target['seed'],
            'expected_epoch':target['expected_epoch'],'checkpoint_path':target['checkpoint_path'],'issues':[]}
        try:
            checkpoint=within(Path(target['checkpoint_path']),root)
            result_path=within(Path(target['result_path']),root)
            if not checkpoint.is_relative_to(root/'baseline_platform/runs'):raise ValueError('Unexpected checkpoint directory')
            before=checkpoint.stat()
            if before.st_size!=target['inventory_checkpoint_bytes']:raise ValueError('Checkpoint size changed since inventory')
            if sha256(result_path)!=target['result_sha256']:raise ValueError('Result JSON changed since inventory')
            result=json.loads(result_path.read_text(encoding='utf-8-sig'))
            row['checkpoint_sha256']=sha256(checkpoint)
            meta,history=inspect_checkpoint(checkpoint,torch,np)
            row['metadata']=meta
            row['issues'],row['ma5_value_at_checkpoint']=assess(meta,history,target,result)
            row['single_epoch_validation_iou']=history[-1]['foreground_iou'] if history else None
            after=checkpoint.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):row['issues'].append('CHECKPOINT_CHANGED_DURING_READ')
            row['status']='PASS_METADATA' if not row['issues'] else 'FAIL_METADATA'
        except Exception as exc:
            row['status']='ERROR';row['issues'].append(type(exc).__name__+': '+str(exc))
        report['results'].append(row)
        print(f"[{i}/66] {target['run']}: {row['status']}",flush=True)
        gc.collect()
    allowed=[root/'baseline_platform'/x for x in ('src','tools','configs','artifacts')]
    bundle=out/'s24_matching_source_bundle.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for entry in manifest['source_files']:
            try:
                path=within(Path(entry['path']),root)
                if not any(path.is_relative_to(folder) for folder in allowed):raise ValueError('Outside approved source/config roots')
                digest=sha256(path)
                if digest!=entry['sha256']:raise ValueError('Source/config changed since inventory')
                relative=path.relative_to(root).as_posix()
                z.write(path,relative)
                report['source_files'].append({'path':entry['path'],'sha256':digest,'bundle_path':relative})
            except Exception as exc:report['source_errors'].append({'path':entry['path'],'error':str(exc)})
        z.writestr('source_manifest.json',json.dumps(report['source_files'],ensure_ascii=False,indent=2))
    passed=sum(x['status']=='PASS_METADATA' for x in report['results'])
    report['summary']={'targets':66,'metadata_pass':passed,'metadata_fail_or_error':66-passed,
        'source_files_bundled':len(report['source_files']),'source_file_errors':len(report['source_errors'])}
    report['validation_inference_verified']=False
    report['source_bundle_sha256']=sha256(bundle)
    dest=out/'s24_checkpoint_metadata.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Report:',dest)
    print('Source bundle:',bundle)
    print(json.dumps(report['summary'],ensure_ascii=False))
    print('No training, model inference, or test data access was performed.')
    return 0 if passed==66 and not report['source_errors'] else 2

if __name__=='__main__':raise SystemExit(main())
