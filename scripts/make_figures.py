import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":"#888","axes.linewidth":0.6,"xtick.color":"#444","ytick.color":"#444","grid.color":"#e6e6e6","grid.linewidth":0.5})
import os; B=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"); R5=B+"/v5/"; R6=B+"/v6/"
R=pd.read_csv(R5+"results_main.csv"); CL=pd.read_csv(R6+"closed_loop.csv")
OUT="figures"; os.makedirs(OUT,exist_ok=True)
C={"blue":"#2a78d6","orange":"#eb6834","aqua":"#1baf7a","yellow":"#eda100","magenta":"#e87ba4","green":"#008300","violet":"#6250d6","gray":"#7a7a74"}
def save(f,n): f.savefig(f"{OUT}/{n}.pdf",bbox_inches="tight"); f.savefig(f"{OUT}/{n}.png",dpi=600,bbox_inches="tight"); plt.close(f)
lab={"GPS-KNN":"GPS (KNN)","GPS":"GPS (GBDT)","Vision(emb)":"Vision (embedding)","Fusion(emb+GPS)":"Vision + GPS","GPS+hist":"GPS + history",
     "Fusion+hist":"Vision + GPS + history","GRU (beam+telemetry seq.)":"GRU (beam + telemetry)","ResNet-18 (fine-tuned)":"ResNet-18",
     "PAR-noImg":"PAR without image","PAR (proposed)":"PAR","ResNet-18 + history (residual)":"ResNet-18 + history"}

# Fig 3 leakage h=0
order=["GPS-KNN","GPS","Vision(emb)","Fusion(emb+GPS)","GPS+hist","Fusion+hist","GRU (beam+telemetry seq.)","ResNet-18 (fine-tuned)","PAR-noImg","PAR (proposed)"]
m=R[R.h==0].groupby(["model","protocol"])["top1"].agg(["mean","std"]).unstack("protocol")
f,ax=plt.subplots(figsize=(4.8,3.2)); y=np.arange(len(order)); hgt=0.36
for k,(p,c,nm) in enumerate([("random",C["blue"],"Random 5-fold split"),("flight",C["orange"],"Flight-disjoint 5-fold split")]):
    v=m["mean"][p].reindex(order).values; e=m["std"][p].reindex(order).values
    ax.barh(y+(k-0.5)*hgt*1.08,v,height=hgt,color=c,xerr=e,error_kw=dict(lw=0.6,capsize=1.5,ecolor="#555"),label=nm)
pers=m["mean"]["flight"]["Persistence"]; ax.axvline(pers,color=C["gray"],ls="--",lw=0.8); ax.text(pers-0.01,-0.75,f"Persistence {pers:.3f}",fontsize=7,color="#444",ha="right")
ax.set_yticks(y); ax.set_yticklabels([lab[o] for o in order]); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel("Top-1 accuracy (h = 0)")
ax.grid(axis="x"); ax.set_axisbelow(True); ax.legend(frameon=False,fontsize=7,ncol=2,loc="upper center",bbox_to_anchor=(0.5,-0.14)); save(f,"fig3_leakage")

# Fig 4 horizon (flight-disjoint)
F=R[R.protocol=="flight"].groupby(["model","h"])["top1"].mean().unstack("h")
series=[("Persistence",C["gray"],"--","o"),("GPS+hist",C["aqua"],"-","^"),("Fusion+hist",C["yellow"],"-","D"),("GRU (beam+telemetry seq.)",C["green"],"-","s"),
        ("ResNet-18 (fine-tuned)",C["magenta"],"-","v"),("ResNet-18 + history (residual)",C["blue"],"-","X"),("PAR-noImg",C["violet"],"-","P"),("PAR (proposed)",C["orange"],"-","o")]
f,ax=plt.subplots(figsize=(4.8,3.0))
for mdl,c,ls,mk in series:
    s=F.loc[mdl].dropna(); ax.plot(s.index,s.values,ls=ls,marker=mk,ms=4.2,lw=1.7 if mdl=="PAR (proposed)" else 1.0,color=c,label=lab.get(mdl,mdl))
ax.set_xticks([0,1,3,5]); ax.set_xticklabels(["0","1 (0.14 s)","3 (0.43 s)","5 (0.71 s)"]); ax.set_xlabel("Prediction horizon h (samples)"); ax.set_ylabel("Top-1 accuracy (flight-disjoint)")
ax.set_ylim(0.3,0.9); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(frameon=False,fontsize=6.5,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.2)); save(f,"fig4_horizon")

# Fig 5 closed loop vs k, clean
cl=CL[CL.degradation=="clean"].groupby(["model","k"])["acc"].mean().unstack("k")
ser=[("Persistence (local sweep)","Persistence",C["gray"],"--","o"),("GPS+hist","GPS + history",C["aqua"],"-","^"),("GRU (beam+telemetry seq.)","GRU",C["green"],"-","s"),
     ("PAR (proposed)","PAR",C["violet"],"-","P"),("ResNet-18 + aug","ResNet-18 (augmented)",C["magenta"],"-","v"),("Fusion PAR x ResNet-18 + aug","PAR-CF (proposed)",C["orange"],"-","o")]
f,ax=plt.subplots(figsize=(4.8,2.8))
for key,nm,c,ls,mk in ser:
    s=cl.loc[key]; ax.plot(s.index,s.values,ls=ls,marker=mk,ms=4.5,lw=1.8 if "Fusion" in key else 1.0,color=c,label=nm)
ov=CL.groupby("k")["overhead"].mean()
ax.set_xticks([1,2,3]); ax.set_xticklabels([f"k = {k}\n({100*ov[k]:.1f}% of full sweep)" for k in (1,2,3)])
ax.set_ylabel("Closed-loop accuracy (clean)"); ax.set_ylim(0,1.02); ax.grid(axis="y"); ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=6.5,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.25)); save(f,"fig5_closed_loop_k")

# Fig 6 closed loop k=2 under degradation
conds=["clean","blur","frame-drop 30%","low-light","occlusion"]; cn={"clean":"Clean","blur":"Blur","frame-drop 30%":"Frame drop\n(30%)","low-light":"Low light","occlusion":"Occlusion"}
mods=[("ResNet-18","ResNet-18",C["blue"]),("ResNet-18 + aug","ResNet-18 (augmented)",C["magenta"]),("PAR (proposed)","PAR",C["violet"]),("Fusion PAR x ResNet-18 + aug","PAR-CF (proposed)",C["orange"])]
d=CL[CL.k==2].groupby(["model","degradation"])["acc"].mean()
f,ax=plt.subplots(figsize=(4.8,2.7)); x=np.arange(len(conds)); w=0.2
for i,(key,nm,c) in enumerate(mods):
    v=[d[(key,cd)] for cd in conds]; xs=x+(i-1.5)*w*1.05
    ax.bar(xs,v,width=w,color=c,label=nm)
    for xi,vi in zip(xs,v):
        if vi<0.3: ax.text(xi,vi+0.015,f"{vi:.2f}",ha="center",fontsize=5.8,color="#333")
ax.set_xticks(x); ax.set_xticklabels([cn[c] for c in conds]); ax.set_ylim(0,1.05); ax.set_ylabel("Closed-loop accuracy (k = 2)")
ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(frameon=False,fontsize=6.5,ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.13)); save(f,"fig6_closed_loop_degraded")
print("ok")
