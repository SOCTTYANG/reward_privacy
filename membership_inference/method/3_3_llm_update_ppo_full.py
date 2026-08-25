import argparse, copy, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from paper_mia import advantages, encode_prompt_responses, gradient_l2, load_records, ppo_clipped_loss, save_json, sequence_logprob

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input_path",required=True); p.add_argument("--base_model",required=True); p.add_argument("--output_path",required=True)
    p.add_argument("--policy_output_dir"); p.add_argument("--max_length",type=int,default=640); p.add_argument("--learning_rate",type=float,default=1e-5)
    p.add_argument("--clip_eps",type=float,default=.2); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    if not 0<a.clip_eps<1: raise ValueError("clip_eps must lie in (0,1)")
    torch.manual_seed(a.seed); data=load_records(a.input_path); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(a.base_model,trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    policy=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,trust_remote_code=True).to(device)
    policy.config.use_cache=False; policy.train()
    for parameter in policy.parameters(): parameter.requires_grad_(True)
    reference=copy.deepcopy(policy).to(device).eval()
    for parameter in reference.parameters(): parameter.requires_grad_(False)
    optimizer=torch.optim.SGD(policy.parameters(),lr=a.learning_rate)
    for step,row in enumerate(tqdm(data,desc="Paper PPO steps"),start=1):
        candidates=row.get("candidate_responses",[])
        responses=[str(c["y_i"]) for c in candidates]; rewards=torch.tensor([float(c["r_i"]) for c in candidates],dtype=torch.float32,device=device)
        if len(responses)!=int(row.get("m",len(responses))) or not responses: raise ValueError(f"record {row.get('id')} has an invalid candidate set")
        adv=advantages(rewards)
        batch=encode_prompt_responses(tok,row["x"],responses,a.max_length,device)
        reference.load_state_dict(policy.state_dict()); reference.eval()
        with torch.no_grad(): ref_logp=sequence_logprob(reference,**batch).detach()
        optimizer.zero_grad(set_to_none=True); new_logp=sequence_logprob(policy,**batch)
        loss=ppo_clipped_loss(new_logp,ref_logp,adv,a.clip_eps); loss.backward()
        norm=gradient_l2(policy.parameters())
        optimizer.step()
        row["advantages"]=[float(x) for x in adv.detach().cpu().tolist()]
        row["ppo_loss"]=float(loss.detach().float().cpu()); row["grad_norm"]=float(norm); row["ppo_step"]=step
    save_json(data,a.output_path)
    if a.policy_output_dir:
        policy.save_pretrained(a.policy_output_dir); tok.save_pretrained(a.policy_output_dir)
    print(f"[DONE] Step 3 wrote {len(data)} records to {a.output_path}")

if __name__=="__main__": main()
