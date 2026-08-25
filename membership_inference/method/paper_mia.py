import json, math, os
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import torch
import torch.nn.functional as F

def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        value = [json.loads(x) for x in f if x.strip()] if path.lower().endswith(".jsonl") else json.load(f)
    if not isinstance(value, list): raise ValueError("Input must be a JSON array or JSONL records")
    return value

def save_json(value: Any, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(value, f, ensure_ascii=False, indent=2)

def first(record: Dict[str, Any], names: Iterable[str]):
    return next((record[n] for n in names if n in record and record[n] is not None), None)

def normalize_triplet(record: Dict[str, Any]) -> Tuple[str, str, str]:
    x = first(record, ("x", "X", "prompt", "instruction", "question", "query"))
    yp = first(record, ("y_plus", "Y_plus", "Y+", "chosen", "better", "preferred"))
    ym = first(record, ("y_minus", "Y_minus", "Y-", "rejected", "worse", "dispreferred"))
    if (yp is None or ym is None) and all(k in record for k in ("response_0", "response_1", "better_response_id")):
        b = int(record["better_response_id"])
        if b not in (0, 1): raise ValueError("better_response_id must be 0 or 1")
        yp, ym = record[f"response_{b}"], record[f"response_{1-b}"]
    if x is None or yp is None or ym is None: raise ValueError(f"Missing triplet in keys {list(record)}")
    return str(x), str(yp), str(ym)

def reward_text(tokenizer, prompt: str, response: str) -> str:
    messages = [{"role":"user","content":str(prompt).strip()}, {"role":"assistant","content":str(response).strip()}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"### Human:\n{messages[0]['content']}\n\n### Assistant:\n{messages[1]['content']}"

@torch.no_grad()
def score_reward_pairs(model, tokenizer, pairs, batch_size, max_length, device):
    out=[]; model.eval()
    for start in range(0, len(pairs), batch_size):
        texts=[reward_text(tokenizer,x,y) for x,y in pairs[start:start+batch_size]]
        batch=tokenizer(texts,padding=True,truncation=True,max_length=max_length,return_tensors="pt")
        logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
        if logits.shape[-1] != 1: raise ValueError("Reward model must output one scalar R(x,y)")
        out.extend(logits.squeeze(-1).detach().float().cpu().tolist())
    return out

def advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 1 or rewards.numel()==0: raise ValueError("rewards must be a non-empty vector")
    return rewards-rewards.mean()

def encode_prompt_responses(tokenizer, prompt: str, responses: Sequence[str], max_length: int, device):
    rows=[]
    for response in responses:
        p=tokenizer(str(prompt),add_special_tokens=True,truncation=True,max_length=max_length)["input_ids"]
        y=tokenizer(str(response),add_special_tokens=False)["input_ids"][:max_length-len(p)]
        if not y: raise ValueError("Prompt leaves no non-empty response to score")
        rows.append((p+y,[-100]*len(p)+y))
    width=max(len(x[0]) for x in rows); pad=tokenizer.pad_token_id
    ids=[]; masks=[]; labels=[]
    for row,label in rows:
        n=width-len(row); ids.append(row+[pad]*n); masks.append([1]*len(row)+[0]*n); labels.append(label+[-100]*n)
    return {"input_ids":torch.tensor(ids,device=device),"attention_mask":torch.tensor(masks,device=device),"labels":torch.tensor(labels,device=device)}

def sequence_logprob(model,input_ids,attention_mask,labels):
    logits=model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).logits[:,:-1]
    target=labels[:,1:]; mask=target.ne(-100); safe=target.masked_fill(~mask,0)
    token=F.log_softmax(logits,dim=-1).gather(-1,safe.unsqueeze(-1)).squeeze(-1)
    return (token*mask).sum(-1)

def ppo_clipped_loss(new_logp,ref_logp,adv,epsilon):
    rho=torch.exp(new_logp-ref_logp); clipped=torch.clamp(rho,1-epsilon,1+epsilon)
    return -torch.minimum(rho*adv,clipped*adv).mean()

def gradient_l2(parameters):
    total=torch.zeros((),dtype=torch.float64)
    for p in parameters:
        if p.requires_grad and p.grad is not None: total += p.grad.detach().double().pow(2).sum().cpu()
    return math.sqrt(total.item())
