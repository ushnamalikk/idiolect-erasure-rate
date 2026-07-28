"""Population accumulation at full length (heavy condition), blogs."""
import json, os, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from src.data import load_blogs_rich, to_xy
from src.rewriters import get_rewriter
CACHE="results/pop400_rewrites.jsonl"
train,test=load_blogs_rich(max_authors=20,load_posts=200,test_slice=(16,31),max_words=400)
items=[(a,i,t) for a,x in test.items() for i,t in enumerate(x)]
print(f"authors={len(train)} test_msgs={len(items)}",flush=True)
cache={}
if os.path.exists(CACHE):
    for l in open(CACHE):
        r=json.loads(l); cache[(r["author"],r["idx"])]=r["rewrite"]
todo=[(a,i,t) for a,i,t in items if (a,i) not in cache]
print(f"cached={len(cache)} todo={len(todo)}",flush=True)
if todo:
    rw=get_rewriter("local",model_name="Qwen/Qwen2.5-1.5B-Instruct",max_new_tokens=600)
    t0=time.time()
    for n,(a,i,t) in enumerate(todo,1):
        o=rw.rewrite([t],"heavy")[0]
        with open(CACHE,"a") as f: f.write(json.dumps({"author":a,"idx":i,"rewrite":o})+"\n")
        cache[(a,i)]=o
        if n%25==0 or n==len(todo):
            el=time.time()-t0; print(f"  {n}/{len(todo)} (~{el/n*(len(todo)-n)/60:.0f} min left)",flush=True)
m=Pipeline([("f",FeatureUnion([
  ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),min_df=2,sublinear_tf=True)),
  ("word",TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=2,sublinear_tf=True))])),
  ("c",LogisticRegression(max_iter=3000,C=10.0,class_weight="balanced"))])
Xtr,ytr=to_xy(train); m.fit(Xtr,ytr)
classes=list(m.named_steps["c"].classes_)
by={}
for a,i,t in items: by.setdefault(a,[]).append((t,cache[(a,i)]))
rng=np.random.default_rng(0); KS=[1,2,3,5,8,12,15]; out={"KS":KS,"orig":[],"assist":[]}
for k in KS:
    for key,use in [("orig",0),("assist",1)]:
        hit=tot=0
        for a,msgs in by.items():
            if len(msgs)<k: continue
            for _ in range(60):
                sel=[msgs[j] for j in rng.choice(len(msgs),k,replace=False)]
                lp=np.log(m.predict_proba([s[use] for s in sel])+1e-12).sum(0)
                hit+= (classes[int(lp.argmax())]==a); tot+=1
        out[key].append(hit/tot)
    print(f"k={k:2d} orig={out['orig'][-1]:.3f} assist={out['assist'][-1]:.3f}",flush=True)
json.dump(out,open("results/population400.json","w"),indent=2)
print("saved -> results/population400.json")
