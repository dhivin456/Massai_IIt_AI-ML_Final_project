"""Cross-part integrity checks; exits non-zero on any acceptance-critical failure."""
from pathlib import Path
import json, sqlite3, io
from contextlib import redirect_stdout
import pandas as pd
from part_a import validate
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; DB=ROOT/'data/instamart.db'
def main():
 with redirect_stdout(io.StringIO()):
  validate()
 required=['customer_segments.csv','lapse_model.joblib','retention_list.csv','triage_results.csv','triage_budget_demo.csv','agent_runs.jsonl','refund_ledger.jsonl','part_a_results.json','part_b_results.json','part_c_results.json','part_d_results.json','policy_eval_overlap10.csv','policy_eval_overlap0.csv']
 missing=[x for x in required if not (OUT/x).exists()]
 assert not missing, f'Missing outputs: {missing}'
 triage=pd.read_csv(OUT/'triage_results.csv',keep_default_na=False); assert len(triage)==164
 seg=pd.read_csv(OUT/'customer_segments.csv'); assert len(seg)==1175 and seg.is_premium.sum()>0
 with sqlite3.connect(DB) as con:
  names={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 assert {'customers','products','orders'} <= names
 a=json.loads((OUT/'part_a_results.json').read_text()); assert a['Q1']['rows']==30 and a['Q2']['rows']==30 and a['Q3']['rows']==11 and a['Q4']['rows']==25 and a['Q5']['rows']==550 and a['Q7']['rows']==183
 b=json.loads((OUT/'part_b_results.json').read_text()); assert b['lapse']['cohort_rows']==1156 and b['lapse']['train_rows']==924 and b['lapse']['test_rows']==232 and b['lapse']['selected_test']['f1_lapsed']>=.55 and b['lapse']['reload_predictions_identical']
 c=json.loads((OUT/'part_c_results.json').read_text()); assert c['few_shot']['classified_json']==135 and c['few_shot']['classified_keyword']==12 and c['few_shot']['flagged_invalid_output']==9 and c['few_shot']['retries']==19 and c['budget_demo']['budget_exhausted']>=1
 d=json.loads((OUT/'part_d_results.json').read_text()); assert d['rag']['overlap10']['answerable_doc_recall']=='12/12' and d['rag']['overlap10']['expected_fact']=='12/12' and d['rag']['overlap10']['unanswerable_fallback']=='2/2'
 expected={'S1':('REFUND',128.25),'S2':('REFUND',135.0),'S3':('ESCALATE',1999.0),'S4':('DENIED',0.0),'S5':('DENIED',0.0),'S6':('NOT_FOUND',0.0),'S7':('FALLBACK_MAX_ITERATIONS',0.0)}
 for row in d['agent_scenarios']:
  status,amount=expected[row['scenario_id']]; assert row['status']==status
  if amount: assert abs(float(row['refund_inr'])-amount)<.01
 assert d['token_limit_demo_status']=='FALLBACK_TOKEN_LIMIT' and d['timeout_demo_status']=='FALLBACK_TIMEOUT'
 assert d['crash_resume']['tools_executed']==['get_order','search_policy','calculate_refund','issue_refund'] and d['crash_resume']['status']=='REFUND'
 print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': main()
