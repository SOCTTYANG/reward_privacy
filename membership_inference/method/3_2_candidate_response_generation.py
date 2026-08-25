import argparse, gc, torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from paper_mia import load_records, save_json, score_reward_pairs

def generation_text(tokenizer,x):
    if getattr(tokenizer,"chat_template",None):
        return tokenizer.apply_chat_template([{"role":"user","content":str(x).strip()}],tokenize=False,add_generation_prompt=True)
    return str(x).strip()

@torch.no_grad()
def generate(model,tokenizer,prompts,m,batch_size,max_prompt_length,max_new_tokens,temperature,top_p,device):
    result=[]; model.eval()
    for start in range(0,len(prompts),batch_size):
        texts=[generation_text(tokenizer,x) for x in prompts[start:start+batch_size]]
        batch=tokenizer(texts,padding=True,truncation=True,max_length=max_prompt_length,return_tensors="pt")
        batch={k:v.to(device) for k,v in batch.items()}; width=batch["input_ids"].shape[1]
        output=model.generate(**batch,do_sample=True,num_return_sequences=m,max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        decoded=tokenizer.batch_decode(output[:,width:],skip_special_tokens=True)
        for i in range(len(texts)):
            values=[x.strip() for x in decoded[i*m:(i+1)*m]]
            if any(not x for x in values): raise ValueError("Generator produced an empty candidate response")
            result.append(values)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input_path",required=True); p.add_argument("--pretrained_llm_path",required=True)
    p.add_argument("--reward_base_model",required=True); p.add_argument("--reward_adapter_path"); p.add_argument("--output_path",required=True)
    p.add_argument("--m",type=int,default=3); p.add_argument("--generation_batch_size",type=int,default=1); p.add_argument("--reward_batch_size",type=int,default=8)
    p.add_argument("--max_prompt_length",type=int,default=512); p.add_argument("--max_new_tokens",type=int,default=128); p.add_argument("--reward_max_length",type=int,default=512)
    p.add_argument("--temperature",type=float,default=.9); p.add_argument("--top_p",type=float,default=.95); a=p.parse_args()
    if a.m<1: raise ValueError("m must be positive")
    data=load_records(a.input_path); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(a.pretrained_llm_path,trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"; model=AutoModelForCausalLM.from_pretrained(a.pretrained_llm_path,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,trust_remote_code=True).to(device)
    candidates=generate(model,tok,[r["x"] for r in data],a.m,a.generation_batch_size,a.max_prompt_length,a.max_new_tokens,a.temperature,a.top_p,device)
    del model,tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    rtok=AutoTokenizer.from_pretrained(a.reward_base_model,trust_remote_code=True)
    if rtok.pad_token_id is None: rtok.pad_token=rtok.eos_token
    rm=AutoModelForSequenceClassification.from_pretrained(a.reward_base_model,num_labels=1,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,trust_remote_code=True)
    if a.reward_adapter_path: rm=PeftModel.from_pretrained(rm,a.reward_adapter_path)
    rm.to(device).eval(); pairs=[(row["x"],y) for row,ys in zip(data,candidates) for y in ys]
    scores=score_reward_pairs(rm,rtok,pairs,a.reward_batch_size,a.reward_max_length,device); cursor=0
    for row,ys in zip(data,candidates):
        row["m"]=a.m; row["candidate_responses"]=[]
        for i,y in enumerate(ys): row["candidate_responses"].append({"candidate_id":i,"y_i":y,"r_i":float(scores[cursor])}); cursor+=1
    save_json(data,a.output_path); print(f"[DONE] Step 2 wrote {len(data)} records to {a.output_path}")

if __name__=="__main__": main()
