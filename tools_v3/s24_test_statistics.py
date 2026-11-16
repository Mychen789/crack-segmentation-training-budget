"""Frozen analysis for fixed-operator, paired-seed and shared-group test scores."""
import itertools
import numpy as np

def summarize(means,ops,delta):
    # means is [budget(low,high), operator]; shared gains cancel by construction.
    low=means[0];high=means[1];gains=high-low
    pairs=[]
    cat=lambda x:1 if x>delta else -1 if x<-delta else 0
    for a,b in itertools.combinations(range(len(ops)),2):
        dl=float(low[a]-low[b]);dh=float(high[a]-high[b]);shift=dh-dl
        pairs.append({'a':ops[a],'b':ops[b],'delta30':dl,'delta90':dh,'shift':shift,
            'changed':abs(shift)>delta,'sign_flip':dl*dh<0,
            'category_change':cat(dl)!=cat(dh),'direct_lead_reversal':cat(dl)*cat(dh)==-1})
    n=len(pairs);changed=sum(p['changed'] for p in pairs)
    return {'n_pairs':n,'n_cross':changed,'Fcross':changed/n,'Mmax':max(abs(p['shift']) for p in pairs),
        'n_sign_flips':sum(p['sign_flip'] for p in pairs),'n_category_changes':sum(p['category_change'] for p in pairs),
        'n_direct_lead_reversals':sum(p['direct_lead_reversal'] for p in pairs),
        'mean_gain':float(gains.mean()),'pairs':pairs}

def analyze(rows,plan):
    ops=plan['operators'];seeds=plan['seeds'];delta=plan['delta']
    indexed={(r['budget'],r['operator'],r['seed']):r for r in rows}
    assert len(indexed)==len(rows)==2*len(ops)*len(seeds)
    gids=sorted(next(iter(indexed.values()))['per_group'])
    values=np.empty((2,len(ops),len(seeds),len(gids)),dtype=float)
    for bi,budget in enumerate((30,90)):
        for oi,op in enumerate(ops):
            for si,seed in enumerate(seeds):
                row=indexed[budget,op,seed];assert sorted(row['per_group'])==gids
                for gi,gid in enumerate(gids):
                    c=row['per_group'][gid];den=c['true_positive']+c['false_positive']+c['false_negative']
                    values[bi,oi,si,gi]=c['true_positive']/den if den else 0.0
                assert abs(values[bi,oi,si].mean()-row['foreground_iou'])<1e-12
    main=summarize(values.mean(axis=(2,3)),ops,delta)
    no_min=[i for i,op in enumerate(ops) if op!='min']
    result={'endpoint':'single checkpoint selected by validation MA(5); group-macro test IoU',
        'primary':main,'without_min':summarize(values[:,no_min].mean(axis=(2,3)),[ops[i] for i in no_min],delta),
        'per_seed':{str(s):summarize(values[:,:,i,:].mean(axis=2),ops,delta) for i,s in enumerate(seeds)},
        'criterion':{'Fcross_at_least':plan['criterion_Fcross'],'met':main['Fcross']>=plan['criterion_Fcross']},
        'operator_mean_scores':{op:{'30':float(values[0,i].mean()),'90':float(values[1,i].mean())} for i,op in enumerate(ops)},
        'groups':len(gids),'seeds':len(seeds),'delta':delta}
    rng=np.random.default_rng(plan['bootstrap_seed']);samples=[]
    ai,bi=np.triu_indices(len(ops),1)
    for _ in range(plan['bootstrap_resamples']):
        ss=rng.integers(0,len(seeds),len(seeds));gg=rng.integers(0,len(gids),len(gids))
        means=values[:,:,ss,:][:,:,:,gg].mean(axis=(2,3))
        gains=means[1]-means[0];shifts=np.abs(gains[ai]-gains[bi])
        samples.append((float(np.mean(shifts>delta)),float(shifts.max())))
    bounds=np.quantile(np.asarray(samples),[.025,.975],axis=0)
    result['paired_seed_shared_group_resampling']={
        'resamples':plan['bootstrap_resamples'],'rng_seed':plan['bootstrap_seed'],
        'scheme':'Resample seed indices and one common group-index list; reuse both for every operator and budget. Operators remain fixed.',
        'Fcross_95_percentile_interval':bounds[:,0].tolist(),'Mmax_95_percentile_interval':bounds[:,1].tolist()}
    return result
