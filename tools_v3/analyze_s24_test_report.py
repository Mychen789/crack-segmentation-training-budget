"""Audit the returned fixed core-test evidence without model or sample access.

Checks frozen identities and independently recomputes the statistics from counts.
Does not import the frozen statistics implementation or load any checkpoint.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'artifacts/s24'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)

def same_number(a,b):
    return math.isfinite(float(a)) and abs(float(a)-float(b))<1e-12

def summary(means,ops,delta):
    category=lambda x:1 if x>delta else -1 if x<-delta else 0
    pairs=[]
    for a,b in itertools.combinations(range(len(ops)),2):
        low=float(means[0,a]-means[0,b]);high=float(means[1,a]-means[1,b])
        shift=high-low
        pairs.append(dict(a=ops[a],b=ops[b],delta30=low,delta90=high,shift=shift,
            changed=abs(shift)>delta,sign_flip=low*high<0,
            category_change=category(low)!=category(high),
            direct_lead_reversal=category(low)*category(high)==-1))
    return dict(n_pairs=len(pairs),n_cross=sum(p['changed'] for p in pairs),
        Fcross=sum(p['changed'] for p in pairs)/len(pairs),
        Mmax=max(abs(p['shift']) for p in pairs),
        n_sign_flips=sum(p['sign_flip'] for p in pairs),
        n_category_changes=sum(p['category_change'] for p in pairs),
        n_direct_lead_reversals=sum(p['direct_lead_reversal'] for p in pairs),
        mean_gain=float(np.mean(means[1]-means[0])),pairs=pairs)

def compare_summary(actual,reported):
    for key in actual:
        if key=='pairs':continue
        require(same_number(actual[key],reported[key]),'Statistic mismatch: '+key)
    require(len(actual['pairs'])==len(reported['pairs']),'Pair count mismatch')
    for p,q in zip(actual['pairs'],reported['pairs']):
        for key,value in p.items():
            require(value==q[key] if isinstance(value,(str,bool)) else same_number(value,q[key]),'Pair mismatch: '+key)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    source=args.bundle.resolve(strict=True)
    expected={'s24_test_confirmation.json','s24_test_execution_freeze.json','s24_frozen_test_plan.json'}
    with zipfile.ZipFile(source) as z:
        require(z.testzip() is None,'ZIP CRC failed')
        require(len(z.namelist())==3 and set(z.namelist())==expected,'Unexpected ZIP entries')
        raw={name:z.read(name) for name in expected}
    plan_raw=(A/'s24_frozen_test_plan.json').read_bytes()
    require(raw['s24_frozen_test_plan.json']==plan_raw,'Pretest specification changed')
    plan=json.loads(plan_raw);ph=sha(plan_raw)
    report=json.loads(raw['s24_test_confirmation.json'])
    frozen=json.loads(raw['s24_test_execution_freeze.json'])
    require(report['status']=='COMPLETE' and not report['errors'],'Batch incomplete or errored')
    require(report['plan_sha256']==frozen['plan_sha256']==ph,'Plan binding failed')
    require(frozen['status']=='CORE_TEST_EXECUTION_FROZEN','Execution record missing')
    require(frozen['authorized_splits']==['test'],'Incorrect access scope')
    require(frozen['execution_files']==plan['execution_files'],'Execution identities differ')
    require(frozen['validation_report_sha256']==sha((A/'s24_validation_reproduction.json').read_bytes()),'Validation identity differs')
    require(report['training_performed'] is False and report['external_test_accessed'] is False,'Scope flag mismatch')
    require(report['test_images_opened'] is True and report['test_labels_opened'] is True,'Actual core-test use unrecorded')
    require(report['test_sample_access_attempted'] is True,'Sample-access record missing')
    times=[datetime.fromisoformat(x) for x in [report['started_utc'],frozen['created_utc'],report['test_access_started_utc'],report['finished_utc']]]
    require(times==sorted(times),'Freeze/access chronology invalid')
    for name,h in plan['execution_files'].items():
        local=(A if name.endswith('.json') else ROOT/'tools_v3')/name
        require(sha(local.read_bytes())==h,'Local audited dependency changed: '+name)
    for key,value in plan['expected_runtime'].items():require(report['runtime'][key]==value,'Runtime mismatch: '+key)
    data=report['test_data']
    require(data['images']==107 and data['groups']==30 and data['pairs_sha256']==plan['pairs_sha256'],'Data identity mismatch')
    targets={t['run']:t for t in plan['targets']}
    rows=report['results'];require(len(rows)==len(targets)==66,'Incomplete target matrix')
    require(len({r['run'] for r in rows})==66 and {r['run'] for r in rows}==set(targets),'Duplicate or wrong models')
    ops=plan['analysis']['operators'];seeds=plan['analysis']['seeds'];delta=plan['analysis']['delta']
    indexed={(r['budget'],r['operator'],r['seed']):r for r in rows}
    require(len(indexed)==66,'Duplicate condition')
    gids=sorted(rows[0]['per_group']);require(len(gids)==30,'Group count mismatch')
    require(sha(json.dumps(gids,ensure_ascii=False).encode())==plan['test_group_ids_sha256'],'Group identity mismatch')
    counts_truth={};max_gap=0.0
    values=np.zeros((2,len(ops),len(seeds),len(gids)),dtype=np.float64)
    for bi,budget in enumerate((30,90)):
        for oi,op in enumerate(ops):
            for si,seed in enumerate(seeds):
                row=indexed[budget,op,seed];target=targets[row['run']]
                require(row['status']=='COMPLETE','Incomplete model')
                for key in ('budget','operator','seed','checkpoint_sha256'):require(row[key]==target[key],'Model identity mismatch: '+key)
                require(row['checkpoint_epoch']==target['expected_epoch'],'Selected epoch mismatch')
                require(sorted(row['per_group'])==gids,'Different group set')
                for gi,gid in enumerate(gids):
                    c=row['per_group'][gid]
                    tp,fp,fn,tn=[c[k] for k in ('true_positive','false_positive','false_negative','true_negative')]
                    require(all(type(v) is int and v>=0 for v in (tp,fp,fn,tn)),'Invalid confusion count')
                    truth=(tp+fn,tn+fp)
                    require(truth==counts_truth.setdefault(gid,truth),'Ground truth differs across models')
                    value=tp/(tp+fp+fn) if tp+fp+fn else 0.0
                    require(same_number(value,c['foreground_iou']),'Group IoU mismatch')
                    values[bi,oi,si,gi]=value
                gap=abs(math.fsum(values[bi,oi,si])/len(gids)-row['foreground_iou']);max_gap=max(max_gap,gap)
                require(gap<1e-12,'Model group-macro IoU mismatch')
    means=values.mean(axis=(2,3));primary=summary(means,ops,delta)
    keep=[i for i,op in enumerate(ops) if op!='min']
    without=summary(means[:,keep],[ops[i] for i in keep],delta)
    per_seed={str(s):summary(values[:,:,si,:].mean(axis=2),ops,delta) for si,s in enumerate(seeds)}
    s=report['statistics'];compare_summary(primary,s['primary']);compare_summary(without,s['without_min'])
    for seed,actual in per_seed.items():compare_summary(actual,s['per_seed'][seed])
    require(s['groups']==30 and s['seeds']==3 and s['delta']==delta,'Statistical specification mismatch')
    require(s['criterion']=={'Fcross_at_least':plan['analysis']['criterion_Fcross'],'met':primary['Fcross']>=plan['analysis']['criterion_Fcross']},'Criterion mismatch')
    for oi,op in enumerate(ops):
        for bi,b in enumerate((30,90)):require(same_number(means[bi,oi],s['operator_mean_scores'][op][str(b)]),'Operator score mismatch')
    # Independent weighted-count implementation of the frozen crossed resampling.
    rng=np.random.default_rng(plan['analysis']['bootstrap_seed'])
    bootstrap=[];pair_indices=list(itertools.combinations(range(len(ops)),2))
    for _ in range(plan['analysis']['bootstrap_resamples']):
        seed_idx=rng.integers(0,len(seeds),len(seeds));group_idx=rng.integers(0,len(gids),len(gids))
        sw=np.bincount(seed_idx,minlength=len(seeds))/len(seeds)
        gw=np.bincount(group_idx,minlength=len(gids))/len(gids)
        sampled=np.einsum('bosg,s,g->bo',values,sw,gw)
        gains=sampled[1]-sampled[0]
        shifts=[abs(gains[a]-gains[b]) for a,b in pair_indices]
        bootstrap.append((sum(x>delta for x in shifts)/len(shifts),max(shifts)))
    bounds=np.quantile(np.asarray(bootstrap),[.025,.975],axis=0)
    resampling=s['paired_seed_shared_group_resampling']
    require(resampling['resamples']==10000 and resampling['rng_seed']==424242,'Resampling specification mismatch')
    require(np.allclose(bounds[:,0],resampling['Fcross_95_percentile_interval'],rtol=0,atol=1e-12),'Fcross interval mismatch')
    require(np.allclose(bounds[:,1],resampling['Mmax_95_percentile_interval'],rtol=0,atol=1e-12),'Mmax interval mismatch')
    # Copy evidence only after all checks; never overwrite a different prior return.
    dest=A/'S24_测试结果回传.zip'
    if dest.exists():require(sha(dest.read_bytes())==sha(source.read_bytes()),'Different prior return bundle exists')
    elif source!=dest:shutil.copy2(source,dest)
    for name in ('s24_test_confirmation.json','s24_test_execution_freeze.json'):
        p=A/name
        if p.exists():require(p.read_bytes()==raw[name],'Different prior evidence exists: '+name)
        else:p.write_bytes(raw[name])
    audit=dict(status='PASS_FROZEN_CORE_TEST_AUDIT',bundle_sha256=sha(source.read_bytes()),
        plan_sha256=ph,report_sha256=sha(raw['s24_test_confirmation.json']),
        execution_freeze_sha256=sha(raw['s24_test_execution_freeze.json']),
        models_verified=66,group_confusion_records_verified=66*30,
        maximum_group_macro_recalculation_gap=max_gap,primary=primary,
        without_min=without,per_seed=per_seed,
        independently_recomputed_intervals={'Fcross':bounds[:,0].tolist(),'Mmax':bounds[:,1].tolist()},
        operator_mean_scores=s['operator_mean_scores'],
        scope='Audit of returned evidence only; no model inference or image/label access on writing device',
        execution_chronology='freeze precedes test access; timing is consistent with the operator-reported uninterrupted run',
        limitations=['Only DeepCrack537/UNet, constant learning rate, fixed operators and three seeds.',
                    'Single validation-selected checkpoints; not test MA(5).',
                    'Descriptive criterion and resampling intervals; no 55-independent-pair inference.',
                    'Runtime/access flags are execution-report evidence, not independent device telemetry.'])
    (A/'s24_test_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table_dir=ROOT/'data/tables'
    with (table_dir/'s24_core_test_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(primary['pairs'][0]))
        writer.writeheader();writer.writerows(primary['pairs'])
    with (table_dir/'s24_core_test_operator_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['operator','test_iou_30','test_iou_90','gain'])
        for oi,op in enumerate(ops):writer.writerow([op,float(means[0,oi]),float(means[1,oi]),float(means[1,oi]-means[0,oi])])
    print(json.dumps({k:v for k,v in audit.items() if k not in ('primary','without_min','per_seed','operator_mean_scores','limitations')},ensure_ascii=False,indent=2))
    print('Primary:',{k:v for k,v in primary.items() if k!='pairs'})

if __name__=='__main__':main()
