import numpy as np, torch, json
from transformers import AutoModel, AutoTokenizer
from src.data import load_blogs_rich
tok=AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD",trust_remote_code=True)
model=AutoModel.from_pretrained("rrivera1849/LUAR-MUD",trust_remote_code=True).eval()
def embed(texts,L,bs=8):
    out=[]
    for i in range(0,len(texts),bs):
        enc=tok(texts[i:i+bs],max_length=L,padding="max_length",truncation=True,return_tensors="pt")
        with torch.no_grad():
            e=model(input_ids=enc["input_ids"].unsqueeze(1),attention_mask=enc["attention_mask"].unsqueeze(1))
        e=e[0] if isinstance(e,(tuple,list)) else e
        e=torch.nan_to_num(e,0.,0.,0.)
        out.append(np.nan_to_num(torch.nn.functional.normalize(e,dim=1).numpy()))
    return np.vstack(out)
tr,te=load_blogs_rich(max_authors=50,load_posts=200,max_words=400)
items=[(a,i,t) for a,x in te.items() for i,t in enumerate(x)]
gold=[a for a,_,_ in items]; orig=[t for _,_,t in items]
for L in [256, 512]:
    authors,profs=[],[]
    for a,x in tr.items():
        authors.append(a); m=embed(x[:60],L).mean(0); profs.append(m/(np.linalg.norm(m)+1e-9))
    P=np.vstack(profs)
    v=embed(orig,L); pred=[authors[i] for i in (v@P.T).argmax(1)]
    print(f"LUAR baseline @full-length, L={L}: {np.mean([p==g for p,g in zip(pred,gold)]):.3f}",flush=True)
