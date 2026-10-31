"""Part A: audit, clean, validate, persist, and query the synthetic exports."""
from pathlib import Path
import json, sqlite3
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; CLEAN=ROOT/'data/clean'; OUT=ROOT/'outputs'; DB=ROOT/'data/instamart.db'
CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)

def audit():
    report={}
    for name in ['customers','products','orders']:
        df=pd.read_csv(RAW/f'{name}.csv')
        report[name]={'shape':list(df.shape),'dtypes':{k:str(v) for k,v in df.dtypes.items()},
          'missing':{k:{'count':int(df[k].isna().sum()),'pct':round(float(df[k].isna().mean()*100),3)} for k in df.columns}}
    o=pd.read_csv(RAW/'orders.csv',keep_default_na=False); c=pd.read_csv(RAW/'customers.csv',keep_default_na=False)
    report['specific']={'raw_orders_rows':len(o),'exact_duplicate_rows':int(o.duplicated().sum()),'pending_rows':int((o.status=='pending').sum()),'distinct_order_id':int(o.order_id.nunique()),'nonnumeric_amounts':int(pd.to_numeric(o.order_amount,errors='coerce').isna().sum()),'negative_delivery_minutes':int(pd.to_numeric(o.delivery_minutes,errors='coerce').lt(0).sum()),'blank_delivery_minutes':int(o.delivery_minutes.astype(str).str.strip().eq('').sum()),'blank_age':int(c.age.astype(str).str.strip().eq('').sum()),'ages_outside_10_100':int(pd.to_numeric(c.age,errors='coerce').dropna().pipe(lambda x: ((x<10)|(x>100)).sum())),'raw_city_spellings':int(c.city.nunique()),'raw_nonblank_gender_spellings':int(c.loc[c.gender.astype(str).str.strip().ne(''),'gender'].nunique()),'blank_gender':int(c.gender.astype(str).str.strip().eq('').sum())}
    (OUT/'audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('AUDIT',json.dumps(report['specific'],sort_keys=True))

def clean():
    c=pd.read_csv(RAW/'customers.csv',keep_default_na=False)
    c['city']=c.city.astype(str).str.strip().str.title()
    c['gender']=c.gender.astype(str).str.strip().str.lower().map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
    c['age']=pd.to_numeric(c.age,errors='coerce'); c.loc[~c.age.between(10,100),'age']=np.nan
    c['age']=c['age'].fillna(c.groupby('city').age.transform('median')).round().astype(int)
    c['signup_date']=pd.to_datetime(c.signup_date).dt.strftime('%Y-%m-%d')
    p=pd.read_csv(RAW/'products.csv')
    o=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    o['order_ts']=pd.to_datetime(o.order_ts)
    # exact duplicates removed before resolving status history
    o=o.drop_duplicates().copy()
    o=o.sort_values(['order_ts','status']).drop_duplicates('order_id',keep='last').copy()
    for col in ['quantity','discount_pct','order_amount','delivery_minutes']:
        o[col]=pd.to_numeric(o[col],errors='coerce')
    o=o.merge(p[['product_id','category','price_inr']],on='product_id',how='left',validate='many_to_one')
    missing=o.order_amount.isna(); o.loc[missing,'order_amount']=(o.loc[missing,'price_inr']*o.loc[missing,'quantity']*(1-o.loc[missing,'discount_pct']/100)).round(2)
    o.loc[o.delivery_minutes<0,'delivery_minutes']=np.nan
    comp=o.status.eq('completed')
    med=o.loc[comp].groupby('category').delivery_minutes.transform('median')
    # align medians by category to all completed rows, based on completed records only
    cat_medians=o.loc[comp].groupby('category').delivery_minutes.median()
    fill=comp & o.delivery_minutes.isna(); o.loc[fill,'delivery_minutes']=o.loc[fill,'category'].map(cat_medians)
    o.loc[o.status.eq('cancelled'),'delivery_minutes']=np.nan
    o['order_ts']=o.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    o=o[['order_id','customer_id','product_id','quantity','order_ts','status','discount_pct','order_amount','delivery_minutes']]
    p['price_inr']=pd.to_numeric(p.price_inr)
    c.to_csv(CLEAN/'customers.csv',index=False); p.to_csv(CLEAN/'products.csv',index=False); o.to_csv(CLEAN/'orders.csv',index=False)
    if DB.exists(): DB.unlink()
    with sqlite3.connect(DB) as con:
        c.to_sql('customers',con,index=False); p.to_sql('products',con,index=False); o.drop(columns=['category'],errors='ignore').to_sql('orders',con,index=False)
    validate(c,p,o)

def validate(c=None,p=None,o=None):
    c=c if c is not None else pd.read_csv(CLEAN/'customers.csv')
    p=p if p is not None else pd.read_csv(CLEAN/'products.csv')
    o=o if o is not None else pd.read_csv(CLEAN/'orders.csv')
    checks={
      'customers_1200':len(c)==1200,'products_40':len(p)==40,'orders_12309':len(o)==12309,
      'unique_order_ids':o.order_id.nunique()==len(o),'no_pending':not o.status.eq('pending').any(),
      'completed_11110':int(o.status.eq('completed').sum())==11110,'cancelled_1199':int(o.status.eq('cancelled').sum())==1199,
      'no_missing_amount':int(o.order_amount.isna().sum())==0,
      'completed_delivery_complete':int(o.loc[o.status.eq('completed'),'delivery_minutes'].isna().sum())==0,
      'cancelled_delivery_missing':int(o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().sum())==1199,
      'completed_revenue':round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2)==2998668.80,
      'six_cities':c.city.nunique()==6,'no_missing_ages':c.age.isna().sum()==0,'gender_counts':c.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30}}
    failed=[k for k,v in checks.items() if not v]
    if failed: raise AssertionError('Checkpoint failures: '+', '.join(failed))
    print('ALL CHECKPOINTS PASSED')

def query_analysis():
    qtext=(ROOT/'sql/queries.sql').read_text()
    queries=[x.strip() for x in qtext.split(';') if x.strip()]
    names=['Q1','Q2','Q3','Q4','Q5','Q6','Q7']; results={}
    with sqlite3.connect(DB) as con:
        for name,q in zip(names,queries):
            df=pd.read_sql_query(q,con); results[name]={'rows':len(df),'head':df.head(10).to_dict('records')}
            if name=='Q2': results[name]['total_revenue']=round(float(df.revenue.sum()),2); results[name]['largest_cell']=df.sort_values('revenue',ascending=False).iloc[0].to_dict()
            if name=='Q5': results[name]['average_total']=round(float(pd.read_sql_query("SELECT AVG(total_spend) a FROM (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id)",con).iloc[0,0]),2)
            if name=='Q6': results[name]['rank_vs_row_number_different']=int((df.spend_rank!=df.row_num).sum())
            if name=='Q7': results[name]['last_cumulative']=float(df.cumulative_revenue.iloc[-1])
    # pandas cross-checks and descriptive statistics
    c=pd.read_csv(CLEAN/'customers.csv'); p=pd.read_csv(CLEAN/'products.csv'); o=pd.read_csv(CLEAN/'orders.csv'); o=o[o.status.eq('completed')].copy()
    pivot=pd.pivot_table(o.drop(columns=['category'],errors='ignore').merge(c[['customer_id','city']],on='customer_id').merge(p[['product_id','category']],on='product_id'),index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
    city=o.merge(c[['customer_id','city']],on='customer_id').groupby('city').order_amount.agg(['sum','mean','count']).round(2)
    s=o.order_amount; q1,q3=s.quantile([.25,.75]); iqr=q3-q1
    stats={'mean':float(s.mean()),'median':float(s.median()),'skew':float(s.skew()),'q1':float(q1),'q3':float(q3),'lower_fence':float(q1-1.5*iqr),'upper_fence':float(q3+1.5*iqr),'outliers_below':int((s<q1-1.5*iqr).sum()),'outliers_above':int((s>q3+1.5*iqr).sum()),'correlation':o[['order_amount','discount_pct','quantity','delivery_minutes']].corr().round(4).to_dict()}
    results['pandas']={'pivot_grand_total':float(pivot.loc['All','All']),'city_aggregates':city.reset_index().to_dict('records'),'stats':stats}
    (OUT/'part_a_results.json').write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ['head','city_aggregates','stats','correlation']} for k,v in results.items() if k.startswith('Q') or k=='pandas'},default=str))

if __name__=='__main__': audit(); clean(); query_analysis()
