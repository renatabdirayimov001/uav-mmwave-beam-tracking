import pandas as pd, numpy as np
import os; B=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"); R5=B+"/v5/"; R6=B+"/v6/"; os.makedirs("tables",exist_ok=True)
R=pd.read_csv(R5+"results_main.csv"); TT=pd.read_csv(R5+"paired_tests.csv"); C=pd.read_csv(R5+"cost.csv")
RB=pd.read_csv(R5+"robustness_summary.csv"); CL=pd.read_csv(R6+"closed_loop.csv"); CT=pd.read_csv(R6+"closed_loop_tests.csv")
H=[0,1,3,5]
NM={"Persistence":"Persistence","GPS-KNN":"GPS (KNN)","GPS":"GPS (GBDT)","GPS+hist":"GPS + history","Vision(emb)":"Vision (embedding)",
    "Fusion(emb+GPS)":"Vision + GPS","Fusion+hist":"Vision + GPS + history","GRU (beam+telemetry seq.)":"GRU (beam + telemetry)",
    "ResNet-18 (fine-tuned)":"ResNet-18","ResNet-18 + augmentation":"ResNet-18 (augmented)","ResNet-18 + history (residual)":"ResNet-18 + history",
    "PAR-noImg":"PAR without image","PAR (proposed)":"\\textbf{PAR}",
    "Abl: PAR features, absolute target":"PAR features, absolute target","Abl: history-only residual":"Residual, beam history only",
    "Abl: PAR without telemetry":"Residual, without telemetry","Abl: PAR non-causal cues (v4)":"Residual, two-sided background",
    "Persistence (local sweep)":"Persistence","ResNet-18":"ResNet-18","ResNet-18 + aug":"ResNet-18 (augmented)",
    "Fusion PAR x ResNet-18":"PAR $\\times$ ResNet-18","Fusion PAR x ResNet-18 + aug":"\\textbf{PAR-CF}"}
def f3(x): return "--" if pd.isna(x) else f"{x:.3f}"
def pfmt(p):
    if pd.isna(p): return "--"
    return "$<$0.001" if p<1e-3 else f"{p:.3f}"
out={}
# leakage
piv=R.groupby(["h","model","protocol"])["top1"].mean().unstack("protocol")
rows=["Persistence","GPS-KNN","GPS","Vision(emb)","Fusion(emb+GPS)","GPS+hist","Fusion+hist","GRU (beam+telemetry seq.)","ResNet-18 (fine-tuned)","PAR-noImg","PAR (proposed)"]
L=[]
for m in rows:
    c=[NM[m]]
    for h in (0,3,5):
        try: r,f=piv.loc[(h,m),"random"],piv.loc[(h,m),"flight"]
        except KeyError: r=f=np.nan
        g="--" if pd.isna(r) else f"{100*(r-f):.1f}".replace("-","$-$")
        c+= [f3(r),f3(f),g]
    L.append(" & ".join(c)+r" \\")
out["leakage"]="\n".join(L)
# main flight table
F=R[R.protocol=="flight"].groupby(["model","h"]).agg(t1=("top1","mean"),sd=("top1","std"),t3=("top3","mean"),pl=("pl_db","mean"))
rows=["Persistence","GPS-KNN","GPS","GPS+hist","Vision(emb)","Fusion(emb+GPS)","Fusion+hist","GRU (beam+telemetry seq.)","ResNet-18 (fine-tuned)","ResNet-18 + augmentation","ResNet-18 + history (residual)","PAR-noImg","PAR (proposed)"]
best={h:F.xs(h,level="h")["t1"].max() for h in H}
L=[]
for m in rows:
    c=[NM[m]]
    for h in H:
        if (m,h) in F.index:
            r=F.loc[(m,h)]; t=f"{r.t1:.3f}"; t=f"\\textbf{{{t}}}" if abs(r.t1-best[h])<1e-9 else t
            c+=[f"{t}$\\pm${r.sd:.3f}", f"{r.t3:.3f} / {r.pl:.2f}"]
        else: c+=["--","--"]
    L.append(" & ".join(c)+r" \\")
out["main"]="\n".join(L)
# ablation
rows=["PAR (proposed)","Abl: PAR features, absolute target","Abl: history-only residual","Abl: PAR without telemetry","PAR-noImg","Abl: PAR non-causal cues (v4)"]
L=[]
for m in rows:
    c=[NM[m].replace("\\textbf{PAR}","PAR (full model)")]
    for h in H:
        t1=F.loc[(m,h),"t1"]
        if m=="PAR (proposed)": c.append(f"{t1:.3f}")
        else:
            r=TT[(TT.h==h)&(TT.vs==m)].iloc[0]; star="$^{*}$" if r.p_holm<0.05 else ""
            c.append(f"{t1:.3f} ({-r.mean_diff_pp:+.1f}{star})".replace("+-","$-$").replace("(-","($-$"))
    L.append(" & ".join(c)+r" \\")
out["ablation"]="\n".join(L)
# wilcoxon main
rows=["Persistence","GPS+hist","Fusion+hist","GRU (beam+telemetry seq.)","ResNet-18 (fine-tuned)","ResNet-18 + history (residual)","PAR-noImg"]
L=[]
for m in rows:
    c=[NM[m]]
    for h in H:
        r=TT[(TT.h==h)&(TT.vs==m)]
        if len(r)==0: c.append("--"); continue
        r=r.iloc[0]; c.append(f"{r.mean_diff_pp:+.1f} ({r.wins}/{r.losses}), {pfmt(r.p_holm)}".replace("+-","$-$").replace("-","$-$") if r.mean_diff_pp<0 else f"+{r.mean_diff_pp:.1f} ({r.wins}/{r.losses}), {pfmt(r.p_holm)}")
    L.append(" & ".join(c)+r" \\")
out["wilcoxon"]="\n".join(L)
# closed loop clean
cl=CL[CL.degradation=="clean"].groupby(["model","k"])[["acc","pl_db","outage_3db"]].mean()
rows=["Persistence (local sweep)","GPS+hist","GRU (beam+telemetry seq.)","PAR-noImg","PAR (proposed)","ResNet-18","ResNet-18 + aug","ResNet-18 + history (residual)","Fusion PAR x ResNet-18","Fusion PAR x ResNet-18 + aug"]
L=[]
for m in rows:
    c=[NM[m]]
    for k in (1,2,3):
        r=cl.loc[(m,k)]; c+=[f"{r.acc:.3f}",f"{r.pl_db:.2f}",f"{100*r.outage_3db:.1f}"]
    L.append(" & ".join(c)+r" \\")
out["closed_clean"]="\n".join(L)
# closed loop degraded k=2,3
conds=["clean","blur","frame-drop 30%","low-light","occlusion"]
d=CL.groupby(["model","k","degradation"])["acc"].mean()
rows=["GPS+hist","GRU (beam+telemetry seq.)","PAR (proposed)","ResNet-18","ResNet-18 + aug","ResNet-18 + history (residual)","Fusion PAR x ResNet-18","Fusion PAR x ResNet-18 + aug"]
L=[]
for m in rows:
    c=[NM[m]]
    for k in (2,3):
        vals=[d[(m,k,cd)] for cd in conds]; c+=[f"{v:.3f}" for v in vals]; c.append(f"{min(vals):.3f}")
    L.append(" & ".join(c)+r" \\")
out["closed_deg"]="\n".join(L)
# closed-loop tests PAR-CF vs, k=2
ct=CT[(CT.ref=="Fusion PAR x ResNet-18 + aug")&(CT.k==2)]
rows=["PAR (proposed)","GRU (beam+telemetry seq.)","GPS+hist","ResNet-18","ResNet-18 + aug","ResNet-18 + history (residual)"]
L=[]
for m in rows:
    c=[NM[m]]
    for cd in conds:
        r=ct[(ct.vs==m)&(ct.degradation==cd)].iloc[0]
        s=f"{r.mean_diff_pp:+.1f} ({r.wins}/{r.losses})"; s=s.replace("-","$-$")
        if r.p_holm<0.05: s="\\textbf{"+s+"}"
        c.append(s)
    L.append(" & ".join(c)+r" \\")
out["closed_tests"]="\n".join(L)
# open-loop robustness (v5) h0,h3
rb=RB.set_index(["h","model"])
rows=["Fusion+hist","ResNet-18 (fine-tuned)","ResNet-18 + augmentation","ResNet-18 + history (residual)","PAR-noImg","PAR (proposed)"]
L=[]
for m in rows:
    c=[NM[m]]
    for h in (0,3):
        if (h,m) not in rb.index: c+=["--"]*5; continue
        r=rb.loc[(h,m)]; c+=[f"{r[x]:.3f}" for x in ["clean","blur","frame-drop 30%","low-light","occlusion"]]
    L.append(" & ".join(c)+r" \\")
out["robust_open"]="\n".join(L)
# cost
cm=C.set_index("model")
rows=[("GPS-KNN","GPS (KNN)"),("GPS","GPS (GBDT)"),("Fusion+hist","Vision + GPS + history$^{a}$"),("PAR-noImg","PAR without image"),("PAR (proposed)","PAR$^{b}$"),
      ("ResNet-18 (fine-tuned)","ResNet-18 / ResNet-18 (augmented)"),("ResNet-18 + history (residual)","ResNet-18 + history")]
L=[]
for k,nm in rows:
    r=cm.loc[k]; g="--" if pd.isna(r.lat_gpu_ms) else f"{r.lat_gpu_ms:.1f}"
    L.append(f"{nm} & {int(r.params_or_nodes):,} {r.unit} & {r.bytes/1e6:.2f} & {r.lat_cpu1_ms:.2f} & {g} \\\\")
cue=cm.loc["image cues (causal background subtraction)","lat_cpu1_ms"]
par=cm.loc["PAR (proposed)"]; rn=cm.loc["ResNet-18 + augmentation"]
L.append(f"\\textbf{{PAR-CF}}$^{{c}}$ & PAR + ResNet-18 & {(par.bytes+rn.bytes)/1e6:.2f} & {par.lat_cpu1_ms+cue+rn.lat_cpu1_ms:.1f} & {rn.lat_gpu_ms:.1f} + {par.lat_cpu1_ms+cue:.1f} (CPU) \\\\")
out["cost"]="\n".join(L)
out["cue_ms"]=f"{cue:.1f}"
for k,v in out.items(): open(f"tables/{k}.tex","w").write(v+"\n")
print(open("tables/ablation.tex").read()); print(open("tables/wilcoxon.tex").read()); print(open("tables/cost.tex").read()); print(open("tables/closed_tests.tex").read())
