"""Historical-match model v2: spaced neighbors, similarity rejection, validation.
Dependencies: numpy and pandas. Source prices are never modified.
"""
import argparse,itertools,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from .baseline import prepare, targets

SETS={'full':['return_5','return_20','volatility_10','volatility_20','drawdown_20','volume_ratio','daily_range','overnight_gap'], 'reduced':['return_5','volatility_20','drawdown_20','volume_ratio','overnight_gap']}
H=10
MIN_MATCHES=5
MATCH_SPACING=11
COOLDOWN=10


def spaced_neighbors(query,z,rows,k,exclude_row=None):
    distances=np.sqrt(np.mean((z-query)**2,axis=1))
    order=np.argsort(distances,kind='stable'); selected=[]
    for j in order:
        if exclude_row is not None and abs(int(rows[j])-int(exclude_row))<MATCH_SPACING: continue
        if all(abs(int(rows[j])-int(rows[a]))>=MATCH_SPACING for a in selected):
            selected.append(int(j))
            if len(selected)==k: break
    selected=np.asarray(selected,dtype=int)
    return selected,distances[selected]


def fit(x,y,ids,config):
    k,history,feature_set,q=config
    rows=ids if history=='all' else ids[-504:]
    cols=SETS[feature_set]
    values=x.iloc[rows][cols].to_numpy(float)
    mean=values.mean(axis=0); scale=values.std(axis=0); scale[scale==0]=1
    z=(values-mean)/scale
    # Training-only support radius from leave-nearby-period-out fifth-neighbor distances.
    reference=[]
    for j in np.linspace(0,len(rows)-1,min(40,len(rows)),dtype=int):
        neighbors,dist=spaced_neighbors(z[j],z,rows,MIN_MATCHES,exclude_row=rows[j])
        if len(dist)>=MIN_MATCHES: reference.append(float(dist[-1]))
    if not reference: raise ValueError('Too little distinct historical support')
    radius=float(np.quantile(reference,q))
    return dict(z=z,rows=rows,labels=y[rows],mean=mean,scale=scale,cols=cols,k=k,radius=radius,config=config,prior=float(y[rows].mean()))


def score(model,x,ids,details=False):
    output=[]
    for i in ids:
        query=(x.iloc[i][model['cols']].to_numpy(float)-model['mean'])/model['scale']
        neighbors,dist=spaced_neighbors(query,model['z'],model['rows'],model['k'])
        keep=dist<=model['radius']; neighbors=neighbors[keep]; dist=dist[keep]
        if len(dist):
            weights=1/(dist+.1)**2; weights/=weights.sum()
            effective=float(1/np.sum(weights**2))
        else: weights=np.array([]); effective=0.
        supported=len(neighbors)>=MIN_MATCHES and effective>=3
        probability=float(weights@model['labels'][neighbors]) if supported else np.nan
        result=dict(row=int(i),score=probability,supported=bool(supported),matches=len(neighbors),effective_matches=effective,baseline=model['prior'])
        if details: result.update(neighbors=neighbors,distances=dist,weights=weights)
        output.append(result)
    return output


def issue_notifications(p,threshold,rows):
    flags=np.zeros(len(rows),bool); last=-10000
    for j,(value,i) in enumerate(zip(p,rows)):
        if np.isfinite(value) and value>=threshold and i-last>=COOLDOWN: flags[j]=True; last=int(i)
    return flags


def event_metrics(rows,y,breach,flags):
    ends=sorted(set(int(breach[i]) for i in rows if y[i]==1)); groups=[]
    for end in ends:
        if not groups or end-groups[-1][-1]>H: groups.append([end])
        else: groups[-1].append(end)
    leads=[]
    for group in groups:
        matches=[int(breach[i]-i) for i,on in zip(rows,flags) if on and y[i]==1 and breach[i] in group]
        if matches: leads.append(max(matches))
    tp=int(sum(on and y[i]==1 for i,on in zip(rows,flags))); fp=int(sum(flags))-tp
    return dict(clusters=len(groups),clusters_warned=len(leads),cluster_recall=len(leads)/len(groups) if groups else 0,notifications=int(sum(flags)),false_notifications=fp,false_notifications_per_252=fp*252/len(rows),notification_precision=tp/(tp+fp) if tp+fp else None,median_lead_sessions=float(np.median(leads)) if leads else None)


def choose_threshold(p,rows,y,breach,beta,budget,fraction):
    best=None
    for t in np.r_[np.linspace(.05,.95,37),1.000001]:
        flags=issue_notifications(p,t,rows); m=event_metrics(rows,y,breach,flags)
        if 0<m['notifications']<3: continue
        if m['false_notifications_per_252']>budget or np.mean(np.nan_to_num(p,nan=-1)>=t)>fraction: continue
        precision=m['notification_precision'] or 0; recall=m['cluster_recall']
        f=(1+beta**2)*precision*recall/max(beta**2*precision+recall,1e-12)
        key=(f,-m['false_notifications'],float(t))
        if best is None or key>best[0]: best=(key,float(t),m)
    return best[1],best[2]


def metrics(scores,threshold,y,breach):
    rows=np.array([s['row'] for s in scores]); p=np.array([s['score'] for s in scores]); covered=np.isfinite(p); truth=y[rows]
    alerts=covered&(p>=threshold); tp=int(sum(alerts&(truth==1))); fp=int(sum(alerts&(truth==0))); fn=int(sum(~alerts&(truth==1)))
    fallback=np.array([s['baseline'] for s in scores]); total=np.where(covered,p,fallback)
    positive=p[covered&(truth==1)]; negative=p[covered&(truth==0)]
    pairs=positive[:,None]-negative[None,:]
    return dict(forecast_days=len(rows),supported_days=int(covered.sum()),coverage=float(covered.mean()),abstained_days=int((~covered).sum()),covered_accuracy=float(np.mean((p[covered]>=threshold)==truth[covered])) if covered.any() else None,alert_policy_accuracy=float(np.mean(alerts==truth)),precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,alert_days=int(alerts.sum()),false_alert_days=fp,covered_auc=float(np.mean((pairs>0)+.5*(pairs==0))) if pairs.size else None,brier_with_explicit_prior_fallback=float(np.mean((total-truth)**2)),**event_metrics(rows,y,breach,issue_notifications(p,threshold,rows)))


def save_model(path,m,dates,thresholds):
    np.savez(path,z=m['z'],rows=m['rows'],labels=m['labels'],mean=m['mean'],scale=m['scale'],cols=np.array(m['cols']),k=m['k'],radius=m['radius'],prior=m['prior'],dates=dates[m['rows']],thresholds=np.array([thresholds[p] for p in ['precision','balanced','detection']]))


def load_model(path):
    with np.load(path) as m:
        model={k:m[k].copy() for k in ['z','rows','labels','mean','scale']}
        model.update(cols=m['cols'].tolist(),k=int(m['k']),radius=float(m['radius']),prior=float(m['prior']))
        return model,m['dates'].copy(),dict(zip(['precision','balanced','detection'],m['thresholds'].tolist()))


def latest_files(out,m,dates,x,d,thresholds):
    item=score(m,x,[len(d)-1],details=True)[0]
    result=dict(date=str(d.Date.iloc[-1].date()),target='APP 10% decline within next ten observed closes',status='supported' if item['supported'] else 'insufficient_historical_similarity',weighted_historical_frequency=float(item['score']) if item['supported'] else None,matches=item['matches'],effective_matches=item['effective_matches'],thresholds={p:None if t>1 else t for p,t in thresholds.items()},alert_conditions={p:bool(item['score']>=t) if item['supported'] else None for p,t in thresholds.items()},note='Not a calibrated probability; insufficient similarity is not a no-shock prediction. Notification delivery/cooldown state is not maintained by this scoring command.')
    (out/'latest_prediction.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame({'historical_date':dates[item['neighbors']],'distance':item['distances'],'weight':item['weights'],'followed_by_shock':m['labels'][item['neighbors']].astype(int)}).to_csv(out/'similar_periods.csv',index=False)
    return result


def main(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    d,x=prepare(args.data); dates=d.Date.dt.strftime('%Y-%m-%d').to_numpy(dtype=str)
    if args.predict:
        m,training_dates,thresholds=load_model(args.predict)
        if dates[-1]<=max(training_dates): raise ValueError('Prediction must follow training period')
        print(json.dumps(latest_files(out,m,training_dates,x,d,thresholds),indent=2)); return
    y,loss=targets(d['Adj Close'],H,.1)
    breach=np.full(len(d),-1,dtype=int)
    for i in range(len(d)-H):
        hits=np.flatnonzero(d['Adj Close'].iloc[i+1:i+11].to_numpy()/d['Adj Close'].iloc[i]-1<=-.1)
        if len(hits): breach[i]=i+1+int(hits[0])
    ids=np.flatnonzero(x.notna().all(axis=1)&np.isfinite(y)); boundary=int(len(ids)*.8)
    if boundary<1000:
        raise ValueError('This fixed research validation design requires at least 1250 eligible labeled rows; supply longer history.')
    test=ids[boundary:]; train_final=ids[:boundary-H]; policy=ids[boundary-190:boundary-H]
    selection_blocks=[ids[420:540],ids[660:780]]
    assert selection_blocks[-1][-1]+H<policy[0] and policy[-1]+H<test[0]
    configs=list(itertools.product([10,20,30],['all','recent_504'],['full','reduced'],[.9,.95]))
    validation=[]
    for config in configs:
        total=[]
        for block in selection_blocks:
            train=ids[ids+H<block[0]]; m=fit(x,y,train,config); total+=score(m,x,block)
        p=np.array([s['score'] for s in total]); rows=np.array([s['row'] for s in total]); baseline=np.array([s['baseline'] for s in total]); supported=np.isfinite(p)
        # Penalize unsupported forecasts explicitly via the train-only prior, not by hiding them.
        loss=float(np.mean((np.where(supported,p,baseline)-y[rows])**2))
        validation.append(dict(neighbors=config[0],history=config[1],feature_set=config[2],radius_quantile=config[3],coverage=float(supported.mean()),brier_with_prior_fallback=loss,config=config))
    eligible=[r for r in validation if r['coverage']>=.5]
    winner=min(eligible or validation,key=lambda r:r['brier_with_prior_fallback']); config=winner['config']
    policy_train=ids[ids+H<policy[0]]; policy_model=fit(x,y,policy_train,config); policy_scores=score(policy_model,x,policy); p=np.array([s['score'] for s in policy_scores])
    thresholds={}; policy_audit={}
    for name,beta,budget,fraction in [('precision',.5,12,.2),('balanced',1.,24,.4),('detection',2.,48,.7)]:
        thresholds[name],policy_audit[name]=choose_threshold(p,policy,y,breach,beta,budget,fraction)
    m=fit(x,y,train_final,config); scores=score(m,x,test)
    comparison={name:metrics(scores,t,y,breach) for name,t in thresholds.items()}
    # Exact old model on identical origins, plus identical cooldown for notification comparisons.
    old_values=x.iloc[train_final][SETS['full']].to_numpy(float)
    old_mean=old_values.mean(axis=0); old_scale=old_values.std(axis=0); old_scale[old_scale==0]=1
    old=dict(mean=old_mean,scale=old_scale,training=(old_values-old_mean)/old_scale,labels=y[train_final],features=np.array(SETS['full']))
    baseline=[]
    for i in test:
        query=(x.iloc[i][old['features'].tolist()].to_numpy()-old['mean'])/old['scale']; distance=np.sqrt(np.mean((old['training']-query)**2,axis=1)); nn=np.argsort(distance,kind='stable')[:30]
        baseline.append(dict(row=int(i),score=float(old['labels'][nn].mean()),baseline=float(old['labels'].mean())))
    comparison['original_basic']=metrics(baseline,.5,y,breach)
    comparison['updated_fixed_50']=metrics(scores,.5,y,breach)
    # Fair threshold-only comparator: same preceding policy window and alert budgets.
    full=x.iloc[policy_train][SETS['full']].to_numpy(float)
    old_mean=full.mean(axis=0); old_scale=full.std(axis=0); old_scale[old_scale==0]=1
    old_z=(full-old_mean)/old_scale; old_policy=[]
    for i in policy:
        query=(x.iloc[i][SETS['full']].to_numpy()-old_mean)/old_scale
        distance=np.sqrt(np.mean((old_z-query)**2,axis=1)); nn=np.argsort(distance,kind='stable')[:30]
        old_policy.append(float(y[policy_train[nn]].mean()))
    original_thresholds={}
    for name,beta,budget,fraction in [('precision',.5,12,.2),('balanced',1.,24,.4),('detection',2.,48,.7)]:
        t,_=choose_threshold(np.array(old_policy),policy,y,breach,beta,budget,fraction)
        original_thresholds[name]=t
        comparison['original_retuned_'+name]=metrics(baseline,t,y,breach)
    save_model(out/'model_v2.npz',m,dates,thresholds)
    pd.DataFrame([{k:v for k,v in r.items() if k!='config'} for r in validation]).to_csv(out/'validation_candidates.csv',index=False)
    export=[]
    for s in scores:
        i=s['row']; export.append(dict(date=dates[i],row=i,actual=int(y[i]),supported=s['supported'],weighted_frequency=s['score'],matches=s['matches'],effective_matches=s['effective_matches'],fallback_prior=s['baseline']))
    pd.DataFrame(export).to_csv(out/'test_predictions.csv',index=False)
    latest=latest_files(out,m,dates[m['rows']],x,d,thresholds)
    results=dict(target='APP 10% decline within ten observed daily closes',selected_configuration=winner,thresholds=thresholds,original_thresholds=original_thresholds,policy_selection=policy_audit,comparison=comparison,split=dict(selection_blocks=[dict(start=dates[b[0]],end=dates[b[-1]]) for b in selection_blocks],policy_start=dates[policy[0]],policy_end=dates[policy[-1]],test_start=dates[test[0]],test_end=dates[test[-1]],training_end=dates[train_final[-1]]),latest=latest,limitations=['Repeatedly inspected retrospective test','Abstentions may include missed shocks; do not interpret as no risk','Spaced neighbors have nonoverlapping target windows but are not necessarily independent events','No S&P 500 or external event/text inputs','Weighted historical frequency is not calibrated probability'])
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/APP.csv'); p.add_argument('--out',default='artifacts/v2'); p.add_argument('--predict'); main(p.parse_args())


