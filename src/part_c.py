"""Part C: privacy-preserving, resilient ticket classification and routing."""
from pathlib import Path
import csv,json,re,random,time
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
MAX_INPUT_TOKENS=400; BASE_DELAY=.05; TOKEN_BUDGET_DEFAULT=100000
PATTERNS=[('email',re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '[EMAIL]'),('phone',re.compile(r'\b[6-9]\d{9}\b'),'[PHONE]'),('order_id',re.compile(r'ORD-\d{6}'),'[ORDER_ID]')]
ZERO_SHOT='''You are a CLASSIFY support-ticket classifier. Allowed categories: billing, delivery, technical, fraud. Return JSON only in the form {"category": "...", "reason": "..."}. The category must be exactly one of the four allowed categories. Do not include prose outside JSON.'''
FEW_SHOT='''You are a CLASSIFY support-ticket classifier. Allowed categories: billing, delivery, technical, fraud. Return JSON only in the form {"category": "...", "reason": "..."}. The category must be exactly one of the four allowed categories. Do not include prose outside JSON.
EXAMPLES:
Ticket: Someone hacked my account; this is fraud.
Reply: {"category": "fraud", "reason": "unauthorized account access"}
Ticket: The app crashes and shows an error.
Reply: {"category": "technical", "reason": "app error"}
Ticket: I was charged twice and need a refund.
Reply: {"category": "billing", "reason": "duplicate charge"}
Ticket: The rider has not delivered my order.
Reply: {"category": "delivery", "reason": "late delivery"}'''
ALLOWED={'billing','delivery','technical','fraud'}

def redact(text):
 counts={}
 out=text or ''
 for name,pat,placeholder in PATTERNS:
  out,n=pat.subn(placeholder,out); counts[name]=n
 return out,counts

def pii_remains(text): return any(p.search(text or '') for _,p,_ in PATTERNS)

def call_retry(messages,ticket_id,log=True):
 retries=0
 while True:
  try: return mock_llm.chat(messages),retries
  except mock_llm.TransientLLMError:
   if retries>=3: raise
   wait=BASE_DELAY*(2**retries)*random.uniform(.9,1.15)
   if log: print(f'RETRY ticket_id={ticket_id} retry={retries} wait={wait:.4f}s')
   time.sleep(wait); retries+=1

def classify_reply(content):
 try:
  parsed=json.loads(content)
  if isinstance(parsed,dict) and parsed.get('category') in ALLOWED:
   return parsed['category'],parsed.get('reason',''),'classified_json'
  return '', '', 'flagged_invalid_output'
 except (json.JSONDecodeError,TypeError):
  words=re.findall(r'[a-z]+',(content or '').lower()); found=ALLOWED.intersection(words)
  if len(found)==1: return next(iter(found)), 'extracted from validated keyword reply', 'classified_keyword'
  return '', '', 'flagged_invalid_output'

def run_pipeline(prompt_name='few_shot',token_budget=None,reset=True,log_retries=True):
 if reset: mock_llm.reset()
 random.seed(42)
 system=FEW_SHOT if prompt_name=='few_shot' else ZERO_SHOT
 tickets=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
 results=[]; total_tokens=0; redaction_totals={'email':0,'phone':0,'order_id':0}; calls=0; retries_total=0
 for r in tickets.to_dict('records'):
  raw=r['text']; cleaned,counts=redact(raw)
  for k,v in counts.items(): redaction_totals[k]+=v
  base={'ticket_id':r['ticket_id'],'customer_id':r['customer_id'],'status':'','category':'','reason':'','tokens_used':0,'true_category':r['true_category'],'redacted_text':cleaned}
  if not raw.strip(): base['status']='skipped_empty'; results.append(base); continue
  if mock_llm.est_tokens(raw)>MAX_INPUT_TOKENS: base['status']='flagged_long'; results.append(base); continue
  assert not pii_remains(cleaned),f'PII remained in {r["ticket_id"]}'
  if token_budget is not None and total_tokens>=token_budget:
   base['status']='budget_exhausted'; results.append(base); continue
  messages=[{'role':'system','content':system},{'role':'user','content':cleaned}]
  response,nretry=call_retry(messages,r['ticket_id'],log=log_retries); retries_total+=nretry; calls+=1
  used=int(response.usage.total_tokens); total_tokens+=used; base['tokens_used']=used
  category,reason,status=classify_reply(response.choices[0].message.content)
  base.update(category=category,reason=reason,status=status); results.append(base)
 # Report every redaction across every raw ticket; guard checks are independently asserted.
 assert all(not pii_remains(x['redacted_text']) for x in results)
 df=pd.DataFrame(results)
 classified=df[df.status.isin(['classified_json','classified_keyword'])]
 accuracy=float((classified.category==classified.true_category).mean()) if len(classified) else 0.0
 labels=sorted(ALLOWED)
 confusion=pd.crosstab(pd.Categorical(classified.true_category,categories=labels),pd.Categorical(classified.category,categories=labels),rownames=['true'],colnames=['predicted'],dropna=False).reindex(index=labels,columns=labels,fill_value=0)
 return {'results':df,'summary':{'prompt':prompt_name,'classified':len(classified),'classified_json':int((df.status=='classified_json').sum()),'classified_keyword':int((df.status=='classified_keyword').sum()),'flagged_invalid_output':int((df.status=='flagged_invalid_output').sum()),'skipped_empty':int((df.status=='skipped_empty').sum()),'flagged_long':int((df.status=='flagged_long').sum()),'accuracy_classified':accuracy,'total_tokens':total_tokens,'tokens_per_classified':total_tokens/len(classified) if len(classified) else 0,'calls':calls,'retries':retries_total,'redactions':redaction_totals,'redactions_total':sum(redaction_totals.values()),'confusion':confusion.to_dict()},'confusion_df':confusion}

def route_and_save(run, budget, filename):
 seg=pd.read_csv(OUT/'customer_segments.csv'); seg['is_premium']=seg.is_premium.astype(str).str.lower().eq('true')
 seg_map=seg.set_index('customer_id')[['segment_name','is_premium']].to_dict('index')
 df=run['results']; out=[]
 for r in df.to_dict('records'):
  s=seg_map.get(r['customer_id'],{'segment_name':'','is_premium':False}); cat=r['category']; status=r['status']
  if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: route,priority='human_review','NORMAL'
  elif cat=='fraud': route,priority='security_team','HIGH'
  elif s['is_premium']: route,priority='premium_desk','HIGH'
  else: route,priority='support_queue','NORMAL'
  out.append({'ticket_id':r['ticket_id'],'customer_id':r['customer_id'],'status':status,'category':cat,'segment_name':s['segment_name'] if cat else '', 'priority':priority,'route':route,'tokens_used':r['tokens_used']})
 pd.DataFrame(out).to_csv(OUT/filename,index=False)

def main():
 zero=run_pipeline('zero_shot',None,True,True); few=run_pipeline('few_shot',None,True,True)
 budget=run_pipeline('zero_shot',6000,True,False)
 if not (budget['results'].status=='budget_exhausted').any(): budget=run_pipeline('zero_shot',5000,True,False)
 final=run_pipeline('few_shot',TOKEN_BUDGET_DEFAULT,True,True)
 route_and_save(budget,None,'triage_budget_demo.csv'); route_and_save(final,None,'triage_results.csv')
 # Persist concise, reproducible metrics; no raw or redacted ticket text is included in reports.
 metrics={'zero_shot':zero['summary'],'few_shot':few['summary'],'budget_demo':{'budget':6000 if (budget['results'].status=='budget_exhausted').any() else 5000,'budget_exhausted':int((budget['results'].status=='budget_exhausted').sum()),'total_tokens':budget['summary']['total_tokens']},'final_few_shot':final['summary']}
 (OUT/'part_c_results.json').write_text(json.dumps(metrics,indent=2,default=str),encoding='utf-8')
 zero['confusion_df'].to_csv(OUT/'triage_zero_shot_confusion.csv')
 print(json.dumps(metrics,indent=2,default=str))
if __name__=='__main__': main()
