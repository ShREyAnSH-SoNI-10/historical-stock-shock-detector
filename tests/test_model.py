import unittest
import numpy as np
import pandas as pd
from shock_model.baseline import targets,nearest
from shock_model.model import spaced_neighbors,score,issue_notifications,event_metrics

class ModelTests(unittest.TestCase):
    def test_forward_target(self):
        y,_=targets(pd.Series([100.,100.,89.]+[100.]*20),10,.1)
        self.assertEqual(y[0],1); self.assertTrue(np.isnan(y[-10:]).all())
    def test_current_close_not_future(self):
        y,_=targets(pd.Series([89.]+[100.]*20),10,.1); self.assertEqual(y[0],0)
    def test_nearest(self):
        idx,_=nearest(np.array([0.]),np.array([[0.],[1.],[3.]]),2); self.assertEqual(idx.tolist(),[0,1])
    def test_spacing(self):
        rows=np.arange(100); z=rows.astype(float)[:,None]
        ids,_=spaced_neighbors(np.array([50.]),z,rows,10)
        self.assertTrue(all(abs(rows[a]-rows[b])>=11 for j,a in enumerate(ids) for b in ids[j+1:]))
    def test_exclusion(self):
        rows=np.arange(100); ids,_=spaced_neighbors(np.array([50.]),rows.astype(float)[:,None],rows,5,exclude_row=50)
        self.assertTrue(all(abs(rows[i]-50)>=11 for i in ids))
    def test_weights_and_support(self):
        m=dict(z=np.arange(5.)[:,None],rows=np.arange(5)*11,labels=np.array([1.,0.,0.,0.,0.]),mean=np.array([0.]),scale=np.array([1.]),cols=['a'],k=5,radius=10.,prior=.2)
        a=score(m,pd.DataFrame({'a':[0.]}),[0],details=True)[0]
        self.assertGreater(a['weights'][0],a['weights'][-1]); self.assertFalse(a['supported']); self.assertTrue(np.isnan(a['score']))
    def test_rejection(self):
        m=dict(z=np.arange(5.)[:,None],rows=np.arange(5)*11,labels=np.zeros(5),mean=np.array([0.]),scale=np.array([1.]),cols=['a'],k=5,radius=.01,prior=.2)
        a=score(m,pd.DataFrame({'a':[0.]}),[0])[0]; self.assertEqual(a['matches'],1); self.assertFalse(a['supported'])
    def test_cooldown(self):
        self.assertEqual(np.flatnonzero(issue_notifications(np.ones(25),.5,np.arange(25))).tolist(),[0,10,20])
    def test_abstention(self):
        self.assertEqual(issue_notifications(np.array([np.nan,.8]),.5,np.arange(2)).tolist(),[False,True])
    def test_clusters(self):
        y=np.zeros(30); y[:3]=1; b=np.full(30,-1); b[:3]=[5,5,6]
        m=event_metrics(np.arange(10),y,b,np.array([True]+[False]*9)); self.assertEqual(m['clusters'],1); self.assertEqual(m['clusters_warned'],1)
if __name__=='__main__': unittest.main()
