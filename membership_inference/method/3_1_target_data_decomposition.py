import argparse, random, torch
import sys
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tool.paper_mia import load_records, normalize_triplet, save_json, score_reward_pairs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--member_path",required=True); p.add_argument("--nonmember_path",required=True)
    p.add_argument("--reward_base_model",required=True); p.add_argument("--reward_adapter_path")
    p.add_argument("--output_path",required=True); p.add_argument("--member_size",type=int); p.add_argument("--nonmember_size",type=int)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--batch_size",type=int,default=8); p.add_argument("--max_length",type=int,default=512)
    a=p.parse_args(); rng=random.Random(a.seed); data=[]
    for membership,path,limit in (("member",a.member_path,a.member_size),("nonmember",a.nonmember_path,a.nonmember_size)):
        raw=load_records(path)
        if limit is not None:
            if limit>len(raw): raise ValueError(f"{membership}: requested {limit}, available {len(raw)}")
            raw=rng.sample(raw,limit)
        for source_index,obj in enumerate(raw):
            x,yp,ym=normalize_triplet(obj)
            data.append({"id":len(data),"membership":membership,"source_index":source_index,"x":x,"y_plus":yp,"y_minus":ym})
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(a.reward_base_model,trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForSequenceClassification.from_pretrained(a.reward_base_model,num_labels=1,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,trust_remote_code=True)
    if a.reward_adapter_path: model=PeftModel.from_pretrained(model,a.reward_adapter_path)
    model.to(device).eval()
    rp=score_reward_pairs(model,tok,[(r["x"],r["y_plus"]) for r in data],a.batch_size,a.max_length,device)
    rm=score_reward_pairs(model,tok,[(r["x"],r["y_minus"]) for r in data],a.batch_size,a.max_length,device)
    for row,plus,minus in zip(data,rp,rm): row.update(r_plus=float(plus),r_minus=float(minus),reward_gap=float(plus-minus))
    save_json(data,a.output_path); print(f"[DONE] Step 1 wrote {len(data)} records to {a.output_path}")

if __name__=="__main__": main()
