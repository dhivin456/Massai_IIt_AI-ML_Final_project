"""Part B: RFM segmentation and leakage-conscious lapse prediction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import joblib
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
SNAP=pd.Timestamp('2025-10-01 00:00:00'); CUTOFF=pd.Timestamp('2025-09-01 00:00:00')

def load():
 c=pd.read_csv(ROOT/'data/clean/customers.csv'); o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts']); p=pd.read_csv(ROOT/'data/clean/products.csv'); return c,o,p

def segmentation(c,o,p):
 comp=o[o.status.eq('completed')].copy(); rfm=comp.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
 rfm['recency']=(SNAP-rfm.last_order).dt.days; rfm=rfm.drop(columns='last_order')
 features=['recency','frequency','monetary']; scaler=StandardScaler(); X=scaler.fit_transform(rfm[features])
 scores=[]
 for k in range(2,9):
  km=KMeans(n_clusters=k,n_init=10,random_state=42); labels=km.fit_predict(X); scores.append({'k':k,'inertia':float(km.inertia_),'silhouette':float(silhouette_score(X,labels))})
 k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42); rfm['segment_id']=km.fit_predict(X)
 prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
 premium_id=int(prof.loc[prof.mean_monetary.idxmax(),'segment_id'])
 # Business labels are assigned from the measured profiles, not assumed cluster ordering.
 names={}
 for row in prof.itertuples():
  if row.segment_id==premium_id: names[row.segment_id]='Premium High Value'
  elif row.mean_frequency>=prof.mean_frequency.quantile(.75): names[row.segment_id]='Frequent Essentials'
  elif row.mean_recency>=prof.mean_recency.median(): names[row.segment_id]='At-Risk / Dormant'
  else: names[row.segment_id]='Growing Regulars'
 rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
 rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(OUT/'customer_segments.csv',index=False)
 pca=PCA(n_components=2); pca.fit(X)
 return rfm,prof,scores,pca.explained_variance_ratio_.tolist(),premium_id,names

def lapse(c,o,p,segments):
 # The feature matrix uses ONLY final-status orders strictly before the cutoff.
 pre=o[o.order_ts<CUTOFF].copy(); comp=pre[pre.status.eq('completed')].copy(); cancelled=pre[pre.status.eq('cancelled')].copy()
 cohort=comp.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean')).reset_index()
 cohort['recency']=(CUTOFF-cohort.last_order).dt.days; cohort=cohort.drop(columns='last_order')
 all_final=pre[pre.status.isin(['completed','cancelled'])].groupby('customer_id').size(); canc=cancelled.groupby('customer_id').size()
 cohort['cancellation_rate']=(canc.reindex(cohort.customer_id,fill_value=0).to_numpy()/all_final.reindex(cohort.customer_id,fill_value=0).to_numpy())
 # Most frequent category from completed pre-cutoff purchases, deterministic tie-break by lexical category.
 cat=comp.drop(columns=['category'],errors='ignore').merge(p[['product_id','category']],on='product_id').groupby(['customer_id','category']).size().rename('n').reset_index().sort_values(['customer_id','n','category'],ascending=[True,False,True]).drop_duplicates('customer_id')
 cohort=cohort.merge(cat[['customer_id','category']].rename(columns={'category':'favourite_category'}),on='customer_id',how='left')
 cohort=cohort.merge(c[['customer_id','city','age','gender','signup_date']],on='customer_id',how='left')
 cohort['tenure_days']=(CUTOFF-pd.to_datetime(cohort.signup_date)).dt.days; cohort=cohort.drop(columns='signup_date')
 sept=o[(o.order_ts>=CUTOFF)&(o.order_ts<pd.Timestamp('2025-10-01'))&o.status.eq('completed')]
 active=set(sept.customer_id); cohort['lapsed']=(~cohort.customer_id.isin(active)).astype(int)
 # Deterministic imputations are computed on the pre-cutoff cohort only; then no missing values remain.
 for col in ['avg_discount','avg_delivery','cancellation_rate','favourite_category','city','gender','age']:
  if col in ['favourite_category','city','gender']: cohort[col]=cohort[col].fillna(cohort[col].mode().iloc[0] if not cohort[col].mode().empty else 'unknown')
  else: cohort[col]=cohort[col].fillna(cohort[col].median())
 cohort['age']=cohort.age.astype(float)
 feature_cols=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','tenure_days','age','city','gender','favourite_category']
 X=cohort[feature_cols]; y=cohort.lapsed
 assert not X.isna().any().any(), 'Missing feature values remain'
 X_train,X_test,y_train,y_test,idx_train,idx_test=train_test_split(X,y,cohort.index,test_size=.2,stratify=y,random_state=42)
 numeric=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','tenure_days','age']; categorical=['city','gender','favourite_category']
 prep=ColumnTransformer([('num',StandardScaler(),numeric),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical)])
 log_pipe=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,random_state=42))])
 rf_pipe=Pipeline([('prep',clone(prep)),('model',RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1))])
 log_grid=GridSearchCV(log_pipe,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1); log_grid.fit(X_train,y_train)
 rf_grid=GridSearchCV(rf_pipe,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1); rf_grid.fit(X_train,y_train)
 def metrics(model):
  pred=model.predict(X_test); return {'confusion_matrix':confusion_matrix(y_test,pred,labels=[0,1]).tolist(),'precision_lapsed':precision_score(y_test,pred,zero_division=0),'recall_lapsed':recall_score(y_test,pred,zero_division=0),'f1_lapsed':f1_score(y_test,pred,zero_division=0),'accuracy':accuracy_score(y_test,pred)}
 baseline=np.zeros(len(y_test),dtype=int)
 # Choose the stronger cross-validated model, with forest winning ties; the brief expects both tuned models.
 selected=rf_grid.best_estimator_ if rf_grid.best_score_>=log_grid.best_score_ else log_grid.best_estimator_
 selected_name='random_forest' if selected is rf_grid.best_estimator_ else 'logistic_regression'
 # class balance: retain all originals and duplicate minority class from TRAINING ONLY.
 train_df=X_train.copy(); train_df['_y']=y_train.to_numpy(); counts=train_df['_y'].value_counts(); majority=int(counts.max()); rng=np.random.RandomState(42); additions=[]
 for cls in [0,1]:
  need=majority-int(counts.get(cls,0))
  if need>0:
   pool=train_df[train_df._y==cls]; sampled=pool.iloc[rng.choice(len(pool),size=need,replace=True)]; additions.append(sampled)
 balanced=pd.concat([train_df]+additions,ignore_index=True); X_bal=balanced.drop(columns='_y'); y_bal=balanced['_y']
 balanced_log=clone(log_grid.best_estimator_); balanced_log.fit(X_bal,y_bal)
 # Refit selected model is the grid-search fitted estimator; persist and prove reload equivalence.
 joblib.dump(selected,OUT/'lapse_model.joblib'); reloaded=joblib.load(OUT/'lapse_model.joblib')
 pred_orig=selected.predict(X_test); pred_reload=reloaded.predict(X_test); assert np.array_equal(pred_orig,pred_reload)
 # Premium list is explicitly post-hoc illustrative; model features remain pre-cutoff only.
 prob=selected.predict_proba(X_test)[:,list(selected.classes_).index(1)]
 test_rows=cohort.loc[idx_test,['customer_id','lapsed']].copy(); test_rows['lapse_probability']=prob
 premium_ids=set(segments.loc[segments.is_premium,'customer_id'])
 ret=test_rows[test_rows.customer_id.isin(premium_ids)].sort_values('lapse_probability',ascending=False).head(10).rename(columns={'lapsed':'actually_lapsed'})
 ret[['customer_id','lapse_probability','actually_lapsed']].to_csv(OUT/'retention_list.csv',index=False)
 # Feature importances and logistic coefficient signs (logistic grid-search model).
 rf=rf_grid.best_estimator_; names_out=rf.named_steps['prep'].get_feature_names_out(); importances=rf.named_steps['model'].feature_importances_
 important=sorted([{'feature':str(a),'importance':float(b)} for a,b in zip(names_out,importances)],key=lambda z:z['importance'],reverse=True)[:5]
 log=log_grid.best_estimator_; coef_names=log.named_steps['prep'].get_feature_names_out(); coefs=log.named_steps['model'].coef_[0]
 coefficient_signs=sorted([{'feature':str(a),'coefficient':float(b),'sign':'positive' if b>0 else 'negative' if b<0 else 'zero'} for a,b in zip(coef_names,coefs)],key=lambda z:abs(z['coefficient']),reverse=True)[:10]
 # baseline frequency scaler mean vs cohort mean; scaler sees training rows only.
 train_freq_mean=float(log.named_steps['prep'].named_transformers_['num'].mean_[numeric.index('frequency')])
 result={'cohort_rows':len(cohort),'lapsed_count':int(y.sum()),'lapsed_rate_pct':float(y.mean()*100),'train_rows':len(X_train),'test_rows':len(X_test),'baseline':{'accuracy':accuracy_score(y_test,baseline),'f1_lapsed':f1_score(y_test,baseline,zero_division=0)},'logistic':{'best_params':log_grid.best_params_,'best_cv_f1':log_grid.best_score_,'test':metrics(log_grid.best_estimator_)},'random_forest':{'best_params':rf_grid.best_params_,'best_cv_f1':rf_grid.best_score_,'test':metrics(rf_grid.best_estimator_)},'selected_model':selected_name,'selected_test':metrics(selected),'reload_predictions_identical':bool(np.array_equal(pred_orig,pred_reload)),'balanced_logistic_test':metrics(balanced_log),'balanced_logistic_best_C':log_grid.best_params_,'feature_importances':important,'logistic_coefficient_signs':coefficient_signs,'training_scaler_frequency_mean':train_freq_mean,'full_cohort_frequency_mean':float(cohort.frequency.mean()),'premium_retention_rows':len(ret),'premium_retention_actually_lapsed':int(ret.actually_lapsed.sum()),'features_missing':int(X.isna().sum().sum())}
 return result

def main():
 c,o,p=load(); seg,prof,scores,pca,prem,names=segmentation(c,o,p); results={'kmeans_scores':scores,'chosen_k':4,'segment_profiles':prof.to_dict('records'),'segment_names':names,'premium_segment_id':prem,'pca_explained_variance_ratio':pca}
 lapse_results=lapse(c,o,p,seg); results['lapse']=lapse_results
 (OUT/'part_b_results.json').write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
 print(json.dumps({'kmeans_scores':scores,'segment_profiles':results['segment_profiles'],'pca':pca,'lapse':{k:v for k,v in lapse_results.items() if k not in ['feature_importances','logistic_coefficient_signs']}},indent=2,default=str))
if __name__=='__main__': main()
