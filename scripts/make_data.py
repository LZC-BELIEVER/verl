"""Controlled mixed arithmetic for update/refit correctness, not a benchmark."""
import random
from pathlib import Path
import pandas as pd
p=Path(__file__).parent/'data';p.mkdir(exist_ok=True)
rng=random.Random(7);rows=[]
for i in range(96):
 a,b=rng.randint(10,99),rng.randint(10,99)
 op=['*','*','+','-'][i%4]
 answer=a*b if op=='*' else a+b if op=='+' else a-b
 rows.append({'data_source':'qat_arithmetic','prompt':[{'role':'user','content':f'What is {a} {op} {b}? Give the integer result.'}], 'ability':'math','reward_model':{'style':'rule','ground_truth':str(answer)},'extra_info':{'index':i,'split':'train' if i<80 else 'test'}})
pd.DataFrame(rows[:80]).to_parquet(p/'train.parquet');pd.DataFrame(rows[80:]).to_parquet(p/'test.parquet')
