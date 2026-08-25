import argparse, math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tool.paper_mia import load_artifact, reward_margin, save_artifact, save_json

def auc_with_ties(labels,scores):
    positives=sum(labels); negatives=len(labels)-positives
    if not positives or not negatives: return None
    order=sorted(range(len(scores)),key=lambda i:scores[i]); rank_sum=0.0; rank=1; i=0
    while i<len(order):
        j=i+1
        while j<len(order) and scores[order[j]]==scores[order[i]]: j+=1
        average=(rank+(rank+j-i-1))/2
        rank_sum += average*sum(labels[order[k]] for k in range(i,j)); rank += j-i; i=j
    return (rank_sum-positives*(positives+1)/2)/(positives*negatives)

def metrics(labels,scores,delta):
    pred=[int(s>=delta) for s in scores]
    tp=sum(y==1 and p==1 for y,p in zip(labels,pred)); tn=sum(y==0 and p==0 for y,p in zip(labels,pred))
    fp=sum(y==0 and p==1 for y,p in zip(labels,pred)); fn=sum(y==1 and p==0 for y,p in zip(labels,pred))
    div=lambda x,y:x/y if y else 0.0; precision=div(tp,tp+fp); recall=div(tp,tp+fn)
    return {"delta":delta,"accuracy":div(tp+tn,len(labels)),"precision":precision,"recall":recall,"f1":div(2*precision*recall,precision+recall),"tpr":recall,"fpr":div(fp,fp+tn),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"auc":auc_with_ties(labels,scores)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input_path",required=True); p.add_argument("--output_path",required=True); p.add_argument("--summary_path",required=True)
    p.add_argument("--lambda1",type=float,default=1.0); p.add_argument("--lambda2",type=float,default=1.0); p.add_argument("--delta",type=float,required=True,help="Threshold fixed on an independent calibration set")
    a=p.parse_args()
    if a.lambda1<0 or a.lambda2<0: raise ValueError("lambda1 and lambda2 are non-negative balancing coefficients")
    data,step3_metadata=load_artifact(a.input_path,expected_stage=3); labels=[]; scores=[]
    if not data: raise ValueError("Step 3 output contains no target records")
    for row in data:
        for key in ("r_plus","r_minus","grad_norm","membership"):
            if key not in row: raise ValueError(f"Step 3 record {row.get('id')} lacks {key}")
        if row["membership"] not in ("member","nonmember"): raise ValueError("membership must be member or nonmember")
        gap=reward_margin(float(row["r_plus"]),float(row["r_minus"])); norm=float(row["grad_norm"])
        if not math.isfinite(gap) or not math.isfinite(norm): raise ValueError("Signals must be finite")
        score=a.lambda1*gap-a.lambda2*norm
        row["reward_gap"]=gap; row["membership_score"]=score; row["predicted_membership"]="member" if score>=a.delta else "nonmember"
        labels.append(int(row["membership"]=="member")); scores.append(score)
    report=metrics(labels,scores,a.delta); report.update(lambda1=a.lambda1,lambda2=a.lambda2,decision_rule="member iff I >= delta",threshold_source="independent calibration set")
    metadata={"stage":4,"reward_model":step3_metadata["reward_model"],"policy_model":step3_metadata["policy_model"],"ppo":step3_metadata["ppo"],"membership":{"lambda1":a.lambda1,"lambda2":a.lambda2,"delta":a.delta,"rule":"member iff I >= delta"}}
    save_artifact(data,metadata,a.output_path); save_json(report,a.summary_path)
    print(f"[DONE] Step 4: {report}")

if __name__=="__main__": main()
