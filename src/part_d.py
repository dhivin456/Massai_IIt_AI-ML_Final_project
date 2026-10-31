"""Part D: TF-IDF policy RAG and a resumable, guarded refund tool agent."""
from pathlib import Path
import json, re, time, sqlite3, uuid, sys
from datetime import datetime, timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; POL=ROOT/'data/policies'; RAW=ROOT/'data/raw'; STATE=ROOT/'state'; OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT)); import mock_llm

def load_chunks(overlap=10):
 chunks=[]
 for path in sorted(POL.glob('*.txt')):
  words=path.read_text(encoding='utf-8').split(); step=40-overlap
  start=0
  while start<len(words):
   end=min(start+40,len(words)); chunks.append({'source':path.name,'text':' '.join(words[start:end])})
   if end>=len(words): break
   start+=step
 return chunks

class PolicyRetriever:
 def __init__(self,overlap=10):
  self.chunks=load_chunks(overlap); self.vectorizer=TfidfVectorizer(); self.matrix=self.vectorizer.fit_transform([x['text'] for x in self.chunks])
 def retrieve(self,query,k=3):
  q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix).ravel(); order=sorted(range(len(scores)),key=lambda i:(-scores[i],self.chunks[i]['source'],i))[:k]
  return [{'source':self.chunks[i]['source'],'score':float(scores[i]),'text':self.chunks[i]['text']} for i in order]

def answer(question,retriever=None):
 retriever=retriever or PolicyRetriever(10); docs=retriever.retrieve(question,3)
 context='\n'.join(f"[{d['source']}] {d['text']}" for d in docs)
 messages=[{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied policy context. If the context does not contain the answer, say exactly that you could not find the information in the policy documents.'},{'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
 response=mock_llm.chat(messages)
 return {'answer':response.choices[0].message.content,'sources':list(dict.fromkeys(d['source'] for d in docs)),'retrieved':docs,'tokens_used':response.usage.total_tokens}

def evaluate_rag():
 qa=pd.read_csv(RAW/'policy_eval.csv',keep_default_na=False); overlap=PolicyRetriever(10); no_overlap=PolicyRetriever(0)
 def eval_with(r):
  rows=[]
  for row in qa.to_dict('records'):
   res=answer(row['question'],r); expected=row['expected_doc']; fact=row['expected_fact']
   rows.append({'question':row['question'],'expected_doc':expected,'top3_has_expected_doc':bool(expected!='NONE' and expected in [x['source'] for x in res['retrieved']]),'expected_fact_found':bool(expected!='NONE' and fact in res['answer']),'fallback':res['answer'].startswith('I could not find'),'answer':res['answer'],'sources':res['sources']})
  df=pd.DataFrame(rows); answerable=df.expected_doc.ne('NONE'); unanswerable=~answerable
  return df,{'answerable_doc_recall':f"{int(df.loc[answerable,'top3_has_expected_doc'].sum())}/{int(answerable.sum())}",'expected_fact':f"{int(df.loc[answerable,'expected_fact_found'].sum())}/{int(answerable.sum())}",'unanswerable_fallback':f"{int(df.loc[unanswerable,'fallback'].sum())}/{int(unanswerable.sum())}",'failed_questions':df.loc[answerable & ~df.expected_fact_found,'question'].tolist()}
 df1,s1=eval_with(overlap); df0,s0=eval_with(no_overlap); df1.to_csv(OUT/'policy_eval_overlap10.csv',index=False); df0.to_csv(OUT/'policy_eval_overlap0.csv',index=False)
 return {'overlap10_chunks':len(overlap.chunks),'overlap10_counts':{x:sum(c['source']==x for c in overlap.chunks) for x in sorted(set(c['source'] for c in overlap.chunks))},'overlap10':s1,'overlap0':s0,'sample_top3':overlap.retrieve('refund policy late delivery',3)}

TOOLS_SPEC=[
 {'name':'get_order','description':'Fetch a clean order record from the Part A SQLite database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve relevant policy chunks for a query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate the permitted refund for completed late-delivered orders.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':'number'}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Record an automatically approved refund of INR 1000 or less.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Record a refund above INR 1000 for human approval.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}
]
RETRIEVER=PolicyRetriever(10)

def get_order(order_id):
 with sqlite3.connect(ROOT/'data/instamart.db') as con:
  row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
 if not row: return {'error':'order not found'}
 return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else float(row[3])}
def search_policy(query): return {'results':RETRIEVER.retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes): return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
def _ledger(tool_name,order_id,refund_inr):
 row={'timestamp':datetime.now(timezone.utc).isoformat(),'tool_name':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
 with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
 return {'refund_inr':row['refund_inr']}
def issue_refund(order_id,refund_inr): return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}

def _status(final):
 m=re.match(r'\s*([A-Z_]+):',final or '')
 return m.group(1) if m else 'UNKNOWN'
def _fallback(status,steps):
 summary=f'{status}: Human handoff required; stopped safely after {len(steps)} completed tool step(s).'
 return summary

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None,resume=False):
 start=time.monotonic(); run_id=run_id or f'{scenario_id}-{uuid.uuid4().hex[:8]}'; cp=STATE/f'{run_id}.json'
 if resume:
  state=json.loads(cp.read_text(encoding='utf-8')); messages=state['messages']; tokens_used=state['tokens_used']; steps=state['steps']; executed=state['tools_executed']; llm_calls=state['llm_calls']; user_message=state['user_message']; scenario_id=state['scenario_id']; start=time.monotonic()
 else:
  messages=[{'role':'system','content':'AGENT: Use tools to verify orders and policies. Never issue refunds outside policy.'},{'role':'user','content':user_message}]
  tokens_used=0; steps=[]; executed=[]; llm_calls=0
 def save(step_number):
  cp.write_text(json.dumps({'run_id':run_id,'scenario_id':scenario_id,'user_message':user_message,'messages':messages,'tokens_used':tokens_used,'step_number':step_number,'steps':steps,'tools_executed':executed,'llm_calls':llm_calls},indent=2),encoding='utf-8')
 while llm_calls<max_iterations:
  if time.monotonic()-start>=time_limit_seconds:
   return {'status':'FALLBACK_TIMEOUT','final_answer':_fallback('FALLBACK_TIMEOUT',steps),'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id,'tools_executed':executed}
  response=mock_llm.chat(messages,tools=TOOLS_SPEC); llm_calls+=1; tokens_used+=int(response.usage.total_tokens)
  if tokens_used>token_limit:
   return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':_fallback('FALLBACK_TOKEN_LIMIT',steps),'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id,'tools_executed':executed}
  msg=response.choices[0].message; calls=msg.tool_calls or []
  if not calls:
   final=msg.content or ''; return {'status':_status(final),'final_answer':final,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id,'tools_executed':executed}
  # mock emits one tool call per assistant turn; support the documented list shape.
  assistant_calls=[]
  for tc in calls:
   name=tc.name; args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
   assistant_calls.append({'id':tc.id,'name':name,'arguments':args})
  messages.append({'role':'assistant','content':None,'tool_calls':assistant_calls})
  for tc in assistant_calls:
   name=tc['name']; args=tc['arguments']
   if name not in TOOL_MAP: observation={'error':f'unknown tool {name}'}
   else:
    try: observation=TOOL_MAP[name](**args)
    except Exception as e: observation={'error':f'tool execution failed: {type(e).__name__}'}
   step={'tool_name':name,'arguments':args,'observation':observation}; steps.append(step); executed.append(name)
   messages.append({'role':'tool','tool_call_id':tc['id'],'name':name,'content':json.dumps(observation,sort_keys=True)})
   save(len(steps))
   if crash_after_tools is not None and len(executed)>=crash_after_tools:
    raise RuntimeError('SIMULATED_CRASH_AFTER_CHECKPOINT')
  # state is already durable after every observation
 if llm_calls>=max_iterations:
  return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':_fallback('FALLBACK_MAX_ITERATIONS',steps),'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id,'tools_executed':executed}
 return {'status':'UNKNOWN','final_answer':'UNKNOWN: human handoff required','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id,'tools_executed':executed}

def run_scenarios():
 scenarios=pd.read_csv(RAW/'agent_scenarios.csv',keep_default_na=False); runs=[]
 (OUT/'refund_ledger.jsonl').write_text('',encoding='utf-8')
 for row in scenarios.to_dict('records'):
  mock_llm.reset(); res=run_agent(row['message'],row['scenario_id'])
  res['expected_outcome']=row['expected_outcome']; res['expected_refund_inr']=float(row['expected_refund_inr']); runs.append(res)
 with (OUT/'agent_runs.jsonl').open('w',encoding='utf-8') as f:
  for r in runs: f.write(json.dumps({'scenario_id':r['run_id'].split('-')[0] if '-' in r['run_id'] else r['run_id'],'status':r['status'],'final_answer':r['final_answer'],'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'steps':r['steps']},ensure_ascii=False)+'\n')
 # stop-safely demos
 s1=scenarios.iloc[0]; mock_llm.reset(); token_demo=run_agent(s1.message,'S1-token-demo',token_limit=200)
 mock_llm.reset(); timeout_demo=run_agent(s1.message,'S1-timeout-demo',time_limit_seconds=0)
 # crash immediately after second tool execution, then resume from saved checkpoint
 crash_run_id='S1-crash-resume'; cp=STATE/f'{crash_run_id}.json'; cp.unlink(missing_ok=True); mock_llm.reset()
 try: run_agent(s1.message,'S1',run_id=crash_run_id,crash_after_tools=2)
 except RuntimeError as e:
  if str(e)!='SIMULATED_CRASH_AFTER_CHECKPOINT': raise
 mock_llm.reset(); resumed=run_agent(s1.message,'S1',run_id=crash_run_id,resume=True)
 # resumed.tools_executed is the entire persisted sequence; ensure exact four calls and no repeats.
 assert resumed['status']=='REFUND' and abs(float(re.search(r'INR ([\d.]+)',resumed['final_answer']).group(1))-128.25)<.01
 assert resumed['tools_executed']==['get_order','search_policy','calculate_refund','issue_refund'],resumed['tools_executed']
 return runs,token_demo,timeout_demo,resumed

def main():
 rag=evaluate_rag(); runs,token_demo,timeout_demo,resumed=run_scenarios()
 summary=[]
 for r in runs:
  amt=''
  # Use the actual calculated refund observation, not a threshold mentioned in prose.
  for step in r['steps']:
   if step['tool_name']=='calculate_refund': amt=step['observation'].get('refund_inr','')
  for step in r['steps']:
   if step['tool_name'] in ('issue_refund','escalate_to_human'): amt=step['arguments'].get('refund_inr',amt)
  summary.append({'scenario_id':r['run_id'].split('-')[0],'status':r['status'],'refund_inr':amt,'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'expected_outcome':r['expected_outcome'],'expected_refund_inr':r['expected_refund_inr']})
 result={'rag':rag,'agent_scenarios':summary,'token_limit_demo_status':token_demo['status'],'timeout_demo_status':timeout_demo['status'],'crash_resume':{'status':resumed['status'],'final_answer':resumed['final_answer'],'tools_executed':resumed['tools_executed'],'llm_calls':resumed['llm_calls'],'tokens_used':resumed['tokens_used']}}
 (OUT/'part_d_results.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
 print(json.dumps(result,indent=2,default=str))
if __name__=='__main__': main()
