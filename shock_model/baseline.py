"""Basic historical-pattern shock model. Dependencies: numpy and pandas.
Run: python -m shock_model.baseline --data PATH
A shock is a close at least 10% lower within the next 10 observed sessions.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES=['return_5','return_20','volatility_10','volatility_20','drawdown_20','volume_ratio','daily_range','overnight_gap']

def prepare(path):
    d=pd.read_csv(path,parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    required=['Ticker','Date','Open','High','Low','Close','Adj Close','Volume']
    if not set(required).issubset(d): raise ValueError('Missing required price/volume columns')
    if set(d.Ticker)!={'APP'} or d.Date.duplicated().any(): raise ValueError('Expected APP-only rows with unique dates')
    if d[required].isna().any().any(): raise ValueError('Missing price/volume/date inputs')
    if not (d[['Open','High','Low','Close','Adj Close']]>0).all().all(): raise ValueError('Prices must be positive')
    if not ((d.High>=d[['Open','Close','Low']].max(axis=1))&(d.Low<=d[['Open','Close','High']].min(axis=1))&(d.Volume>=0)).all(): raise ValueError('Invalid OHLC or volume')
    c=d['Adj Close']; r=c.pct_change()
    x=pd.DataFrame({'return_5':c.pct_change(5),'return_20':c.pct_change(20),'volatility_10':r.rolling(10).std(),'volatility_20':r.rolling(20).std(),'drawdown_20':c/c.rolling(20).max()-1,'volume_ratio':d.Volume/d.Volume.rolling(20).mean(),'daily_range':(d.High-d.Low)/d.Close,'overnight_gap':d.Open/d.Close.shift(1)-1})
    return d,x.replace([np.inf,-np.inf],np.nan)

def targets(close,horizon,shock):
    future=pd.concat([close.shift(-i) for i in range(1,horizon+1)],axis=1)
    loss=future.min(axis=1)/close-1
    y=(loss<=-shock).astype(float)
    y[future.isna().any(axis=1)]=np.nan
    return y.to_numpy(),loss.to_numpy()

def nearest(query,training,k):
    distances=np.sqrt(np.mean((training-query)**2,axis=1))
    idx=np.argsort(distances,kind='stable')[:min(k,len(training))]
    return idx,distances[idx]

def evaluate(y,p,threshold):
    alert=p>=threshold; tp=int(np.sum(alert&(y==1))); fp=int(np.sum(alert&(y==0))); fn=int(np.sum(~alert&(y==1)))
    pairs=p[y==1,None]-p[y==0][None,:]
    return dict(forecast_days=len(y),positive_forecast_days=int(y.sum()),prevalence=float(y.mean()),brier_loss=float(np.mean((p-y)**2)),auc=float(np.mean((pairs>0)+.5*(pairs==0))) if pairs.size else None,alert_days=int(alert.sum()),false_alert_days=fp,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)

def main(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    d,x=prepare(args.data)
    if args.predict:
        saved=np.load(args.predict)
        mean,scale,training,labels,dates=saved['mean'],saved['scale'],saved['training'],saved['labels'],saved['dates']
        horizon=int(saved['horizon']); shock=float(saved['shock']); k=int(saved['k']); threshold=float(saved['threshold'])
        train_end=str(saved['train_end'])
    else:
        horizon=args.horizon; shock=args.shock; k=args.neighbors; threshold=.5
        y,loss=targets(d['Adj Close'],horizon,shock)
        ids=np.flatnonzero(x.notna().all(axis=1)&np.isfinite(y))
        boundary=int(len(ids)*.8)
        train=ids[:boundary-horizon]; test=ids[boundary:]
        if len(train)<k or len(test)<30: raise ValueError('Not enough rows for the chronological split')
        assert train[-1]+horizon<test[0]
        mean=x.iloc[train].mean().to_numpy(); scale=x.iloc[train].std(ddof=0).to_numpy().copy(); scale[scale==0]=1
        training=(x.iloc[train].to_numpy()-mean)/scale; labels=y[train]; dates=d.Date.iloc[train].dt.strftime('%Y-%m-%d').to_numpy(dtype=str)
        train_end=str(d.Date.iloc[train[-1]].date())
        probs=[]
        for i in test:
            neighbors,_=nearest((x.iloc[i].to_numpy()-mean)/scale,training,k)
            probs.append(float(labels[neighbors].mean()))
        p=np.array(probs); baseline=np.full(len(test),labels.mean())
        metrics=dict(scope='APP-only historical-pattern experiment',target=f'{shock:.0%} adjusted-close decline within next {horizon} observed sessions',training_rows=len(train),training_start=str(d.Date.iloc[train[0]].date()),training_end=train_end,test_start=str(d.Date.iloc[test[0]].date()),test_end=str(d.Date.iloc[test[-1]].date()),neighbor_count=k,alert_threshold=threshold,test=evaluate(y[test],p,threshold),constant_baseline=evaluate(y[test],baseline,threshold),evaluation_status='Chronological retrospective check; this dataset was already inspected in previous experiments')
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        pd.DataFrame({'date':d.Date.iloc[test].dt.strftime('%Y-%m-%d').to_numpy(),'observed_shock':y[test].astype(int),'historical_shock_frequency':p,'alert':p>=threshold}).to_csv(out/'test_predictions.csv',index=False)
        np.savez(out/'basic_model.npz',mean=mean,scale=scale,training=training,labels=labels,dates=dates,features=np.array(FEATURES),horizon=horizon,shock=shock,k=k,threshold=threshold,train_end=train_end)
        print(json.dumps(metrics,indent=2))
    if x.iloc[-1].isna().any(): raise ValueError('Latest row has incomplete features')
    if args.predict and str(d.Date.iloc[-1].date())<=train_end: raise ValueError('Prediction date must be after the training period')
    neighbors,distances=nearest((x.iloc[-1].to_numpy()-mean)/scale,training,k)
    score=float(labels[neighbors].mean()); count=int(labels[neighbors].sum())
    latest=dict(as_of=str(d.Date.iloc[-1].date()),ticker='APP',decline_threshold=shock,horizon_sessions=horizon,similar_periods=len(neighbors),similar_periods_followed_by_shock=count,historical_shock_frequency=score,alert_threshold=threshold,alert=score>=threshold,mean_standardized_distance=float(distances.mean()),note='Historical neighbor frequency, not a calibrated probability or guarantee; neighboring periods may overlap')
    (out/'latest_prediction.json').write_text(json.dumps(latest,indent=2))
    pd.DataFrame({'historical_date':dates[neighbors],'standardized_distance':distances,'followed_by_shock':labels[neighbors].astype(int)}).to_csv(out/'similar_periods.csv',index=False)
    print(json.dumps(latest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/APP.csv'); p.add_argument('--out',default='artifacts/baseline'); p.add_argument('--shock',type=float,default=.10); p.add_argument('--horizon',type=int,default=10); p.add_argument('--neighbors',type=int,default=30); p.add_argument('--predict',help='Load a saved basic_model.npz instead of training'); args=p.parse_args()
    if not (0<args.shock<1 and args.horizon>=1 and args.neighbors>=1): p.error('Invalid shock, horizon or neighbor count')
    main(args)

