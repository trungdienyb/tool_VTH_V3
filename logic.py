# -*- coding: utf-8 -*-
"""
logic_merged_fix.py — PRO++ (fix KeyError 'w_lr', meta migration-safe)
---------------------------------------------------------------------
- Thêm _ensure_meta() để tự điền các khóa mới (w_lr, w_mlp, w_beta, use_rate...)
  khi load state cũ (ai_state_pro.json) để KHÔNG còn KeyError.
- Các đặc tính chiến lược giữ nguyên theo bản PRO++ trước.
"""

from __future__ import annotations

import json, math, os, random, threading, time
from typing import Dict, List, Optional, Tuple
from itertools import pairwise

STATE_OLD = "ai_state_elite.json"
STATE_PRO = "ai_state_pro.json"
HISTORY_FILE = "history.json"
KNOWLEDGE_FILE = "ai_knowledge.jsonl"
ROOMS = list(range(1,9))
FEATURE_DIM = 20
MAX_HISTORY = 300

_REFRESH_T = None
_REFRESH_STOP = threading.Event()

# -------------------------- utils --------------------------
def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def circ_dist(a: Optional[int], b: Optional[int]) -> int:
    if a is None or b is None: return 4
    a=int(a); b=int(b)
    d = abs(a-b)
    return d if d < 8-d else 8-d

# ------------------------ online scaler --------------------
class OnlineScaler:
    def __init__(self, dim: int):
        self.dim=dim; self.n=0
        self.mean=[0.0]*dim; self.M2=[0.0]*dim
    def partial_fit(self, x: List[float]):
        x=(x+[0.0]*self.dim)[:self.dim]
        self.n+=1; n=self.n
        for i in range(self.dim):
            d=x[i]-self.mean[i]
            self.mean[i]+=d/n
            self.M2[i]+=d*(x[i]-self.mean[i])
    def transform(self, x: List[float]):
        x=(x+[0.0]*self.dim)[:self.dim]
        out=[0.0]*self.dim
        for i in range(self.dim):
            var=self.M2[i]/(self.n-1) if self.n>1 else 0.0
            std=math.sqrt(var) if var>1e-12 else 1.0
            out[i]=(x[i]-self.mean[i])/std
        return out
    def to_dict(self): return {"dim": self.dim, "n": self.n, "mean": self.mean, "M2": self.M2}
    @staticmethod
    def from_dict(d):
        sc=OnlineScaler(d.get("dim", FEATURE_DIM))
        sc.n=int(d.get("n",0)); sc.mean=list(d.get("mean",[0.0]*sc.dim)); sc.M2=list(d.get("M2",[0.0]*sc.dim))
        return sc

# --------------------- light learners ----------------------
class OnlineLogRegAdam:
    def __init__(self, dim: int):
        self.dim=dim; self.w=[0.0]*dim; self.b=0.0
        self.m=[0.0]*dim; self.v=[0.0]*dim; self.mb=0.0; self.vb=0.0; self.t=0
    @staticmethod
    def _sigmoid(z: float)->float:
        if z>=0: ez=math.exp(-z); return 1/(1+ez)
        ez=math.exp(z); return ez/(1+ez)
    def _fit(self, x): return (x+[0.0]*self.dim)[:self.dim]
    def predict_proba(self, x: List[float])->float:
        x=self._fit(x); z=self.b
        for i in range(self.dim): z+=self.w[i]*x[i]
        return self._sigmoid(z)
    def step(self, x: List[float], y: int, lr: float, l2: float, beta1=0.9, beta2=0.999, eps=1e-8, clip=2.0):
        x=self._fit(x); self.t+=1
        p=self.predict_proba(x); p=max(min(p,1-1e-8),1e-8)
        g=p-y; gb=g
        gw=[g*xi + l2*self.w[i] for i,xi in enumerate(x)]
        gnorm=math.sqrt(sum(gi*gi for gi in gw)+gb*gb)
        if gnorm>clip:
            s=clip/(gnorm+1e-12); gw=[gi*s for gi in gw]; gb*=s
        for i in range(self.dim):
            self.m[i]=beta1*self.m[i]+(1-beta1)*gw[i]
            self.v[i]=beta2*self.v[i]+(1-beta2)*(gw[i]*gw[i])
            mhat=self.m[i]/(1-beta1**self.t); vhat=self.v[i]/(1-beta2**self.t)
            self.w[i]-=lr*mhat/(math.sqrt(vhat)+eps)
        self.mb=beta1*self.mb+(1-beta1)*gb; self.vb=beta2*self.vb+(1-beta2)*(gb*gb)
        mbh=self.mb/(1-beta1**self.t); vbh=self.vb/(1-beta2**self.t)
        self.b-=lr*mbh/(math.sqrt(vbh)+eps)
        return -(y*math.log(p)+(1-y)*math.log(1-p))
    def to_dict(self): 
        return {"dim": self.dim, "w": self.w, "b": self.b, "m": self.m, "v": self.v, "mb": self.mb, "vb": self.vb, "t": self.t}
    @staticmethod
    def from_dict(d):
        m=OnlineLogRegAdam(d.get("dim", FEATURE_DIM))
        for k in ["w","b","m","v","mb","vb","t"]:
            setattr(m,k, d.get(k, getattr(m,k)))
        return m

class TinyMLP:
    def __init__(self, in_dim: int, hid: int=16):
        self.in_dim=in_dim; self.hid=hid
        r1=math.sqrt(6/(in_dim+hid)); r2=math.sqrt(6/(hid+1))
        import random as _r
        self.W1=[[ _r.uniform(-r1,r1) for _ in range(in_dim)] for __ in range(hid)]
        self.b1=[0.0]*hid; self.W2=[_r.uniform(-r2,r2) for _ in range(hid)]; self.b2=0.0
        self.mW1=[[0.0]*in_dim for _ in range(hid)]; self.vW1=[[0.0]*in_dim for _ in range(hid)]
        self.mb1=[0.0]*hid; self.vb1=[0.0]*hid; self.mW2=[0.0]*hid; self.vW2=[0.0]*hid; self.mb2=0.0; self.vb2=0.0; self.t=0
    @staticmethod
    def _sigmoid(z: float)->float:
        if z>=0: ez=math.exp(-z); return 1/(1+ez)
        ez=math.exp(z); return ez/(1+ez)
    @staticmethod
    def _relu(x:float)->float: return x if x>0 else 0.0
    def _fit(self,x): return (x+[0.0]*self.in_dim)[:self.in_dim]
    def forward(self,x: List[float], dropout_p: float=0.0, train: bool=False):
        x=self._fit(x); h=[0.0]*self.hid; pre=[0.0]*self.hid; mask=[1.0]*self.hid
        for j in range(self.hid):
            s=self.b1[j]
            W1j=self.W1[j]
            for i in range(self.in_dim): s+=W1j[i]*x[i]
            pre[j]=s; val=self._relu(s)
            if train and dropout_p>0.0:
                import random as _r
                if _r.random()<dropout_p: mask[j]=0.0; val=0.0
                else: val/=(1.0-dropout_p)
            h[j]=val
        z=self.b2
        for j in range(self.hid): z+=self.W2[j]*h[j]
        p=self._sigmoid(z); return p,h,pre,mask
    def step(self, x: List[float], y: int, lr: float, l2: float, beta1=0.9, beta2=0.999, eps=1e-8, clip=2.0, dropout_p=0.15):
        x=self._fit(x); self.t+=1
        p,h,pre,mask=self.forward(x, dropout_p=dropout_p, train=True)
        p=max(min(p,1-1e-8),1e-8); g=p-y; gb2=g
        gW2=[g*hj + l2*self.W2[j] for j,hj in enumerate(h)]
        gh=[g*self.W2[j]*(1.0 if pre[j]>0 else 0.0)*mask[j] for j in range(self.hid)]
        gW1=[[0.0]*self.in_dim for _ in range(self.hid)]; gb1=[0.0]*self.hid
        for j in range(self.hid):
            gb1[j]=gh[j]
            for i in range(self.in_dim):
                gW1[j][i]=gh[j]*x[i] + l2*self.W1[j][i]
        s=gb2*gb2
        for j in range(self.hid):
            s+=gW2[j]*gW2[j]+gb1[j]*gb1[j]
            for i in range(self.in_dim): s+=gW1[j][i]*gW1[j][i]
        gnorm=math.sqrt(s)
        if gnorm>clip:
            c=clip/(gnorm+1e-12); gb2*=c
            for j in range(self.hid):
                gW2[j]*=c; gb1[j]*=c
                for i in range(self.in_dim): gW1[j][i]*=c
        def upd(param, grad, m, v):
            m=beta1*m+(1-beta1)*grad; v=beta2*v+(1-beta2)*(grad*grad)
            mhat=m/(1-beta1**self.t); vhat=v/(1-beta2**self.t)
            param-=lr*mhat/(math.sqrt(vhat)+eps); return param,m,v
        self.b2,self.mb2,self.vb2=upd(self.b2, gb2, self.mb2, self.vb2)
        for j in range(self.hid):
            self.W2[j],self.mW2[j],self.vW2[j]=upd(self.W2[j], gW2[j], getattr(self,'mW2')[j], getattr(self,'vW2')[j])
            self.b1[j],self.mb1[j],self.vb1[j]=upd(self.b1[j], gb1[j], getattr(self,'mb1')[j], getattr(self,'vb1')[j])
            for i in range(self.in_dim):
                self.W1[j][i],self.mW1[j][i],self.vW1[j][i]=upd(self.W1[j][i], gW1[j][i], getattr(self,'mW1')[j][i], getattr(self,'vW1')[j][i])
        return -(y*math.log(p)+(1-y)*math.log(1-p))
    def predict_mc(self, x: List[float], T:int=6, dropout_p:float=0.15)->Tuple[float,float]:
        x=self._fit(x); vals=[]
        import random as _r
        for _ in range(T):
            p,_,_,_=self.forward(x, dropout_p=dropout_p, train=True); vals.append(p)
        m=sum(vals)/max(1,len(vals)); var=sum((v-m)**2 for v in vals)/max(1,len(vals))
        return float(m), float(math.sqrt(var))
    def to_dict(self):
        return {"in_dim": self.in_dim, "hid": self.hid, "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "mW1": getattr(self,'mW1',[[0.0]*self.in_dim for _ in range(self.hid)]),
                "vW1": getattr(self,'vW1',[[0.0]*self.in_dim for _ in range(self.hid)]),
                "mb1": getattr(self,'mb1',[0.0]*self.hid),
                "vb1": getattr(self,'vb1',[0.0]*self.hid),
                "mW2": getattr(self,'mW2',[0.0]*self.hid),
                "vW2": getattr(self,'vW2',[0.0]*self.hid),
                "mb2": getattr(self,'mb2',0.0),
                "vb2": getattr(self,'vb2',0.0),
                "t": self.t}
    @staticmethod
    def from_dict(d):
        net=TinyMLP(d.get("in_dim", FEATURE_DIM), d.get("hid",16))
        for k in ["W1","b1","W2","b2","mW1","vW1","mb1","vb1","mW2","vW2","mb2","vb2","t"]:
            setattr(net,k, d.get(k, getattr(net,k)))
        return net

# --------------------- governor / meta ---------------------
def _ensure_beta(d: Optional[Dict])->Dict[int,float]:
    d = {} if d is None else dict(d); out={}
    for k,v in d.items():
        try: out[int(k)] = float(v)
        except: pass
    for r in ROOMS: out.setdefault(r, 1.0)
    return out

def _default_meta()->Dict:
    return {
        "lr": 0.03, "l2": 1e-4, "dropout": 0.12, "adam_eps": 1e-8, "grad_clip": 2.0,
        "epsilon": 0.04, "epsilon_min": 0.02, "epsilon_max": 0.18,
        "use_rate": False,
        "w_lr": 0.5, "w_mlp": 0.45, "w_beta": 0.05
    }

def _ensure_meta(m: Optional[Dict])->Dict:
    d = _default_meta()
    if isinstance(m, dict):
        # copy in only known keys and keep unknowns (forward compatibility)
        for k, v in m.items():
            d[k] = v
    # harden: make sure ensemble weights exist and sum>0
    s = float(d.get("w_lr", 0.5)) + float(d.get("w_mlp", 0.45)) + float(d.get("w_beta", 0.05))
    if s <= 0:
        d["w_lr"], d["w_mlp"], d["w_beta"] = 0.5, 0.45, 0.05
    return d

def _ensure_governor(g: Optional[Dict])->Dict:
    if g is None: g={}
    g.setdefault("beta_a", {r:1.0 for r in ROOMS})
    g.setdefault("beta_b", {r:1.0 for r in ROOMS})
    g["beta_a"]=_ensure_beta(g["beta_a"]); g["beta_b"]=_ensure_beta(g["beta_b"])
    g.setdefault("chosen_recent", []); g.setdefault("chosen_max", 16)
    g.setdefault("prev_choice", None); g.setdefault("loss_streak", 0)
    g.setdefault("total_wins", 0); g.setdefault("total_losses", 0)
    g.setdefault("win_streak", 0); g.setdefault("best_streak", 0)
    g.setdefault("last_seen_issue", None)
    g.setdefault("cooldown", {r:0 for r in ROOMS})
    g.setdefault("cooldown_decay", 1); g.setdefault("cooldown_init_room", 2); g.setdefault("cooldown_init_neighbor", 1)
    g.setdefault("cooldown_strength", 0.15)
    g.setdefault("protect_mode_enabled", True); g.setdefault("protect_neighbor_radius", 2)
    g.setdefault("protect_penalty", 0.45)
    g.setdefault("last_losses", [])
    g.setdefault("tactics", {"motif_k": 6, "momentum_decay": 0.85, "lookahead_h": 2, "neighbor_band": 1, "max_tactical_pen": 0.55})
    return g

# --------------------- features (rate-less) ----------------
def _win_rate(room: int, seq: List[int], w: int)->float:
    if not seq: return 0.0
    sub = seq[-w:] if len(seq)>=w else seq
    return sub.count(room)/float(len(sub))

def _features(room: int, stats_100: Dict[int,int], seq: List[int], meta: Dict) -> List[float]:
    n=len(seq); total=max(1,n)
    long_total=sum(int(stats_100.get(r,0)) for r in ROOMS) or 1
    long_freq = stats_100.get(room,0)/long_total
    if not meta.get("use_rate", False):
        long_freq = 0.0
    short_freq = seq.count(room)/total if meta.get("use_rate", False) else 0.0

    last1 = seq[-1] if n else None
    d_last1 = float(circ_dist(room, last1))
    w3,w5,w7 = _win_rate(room, seq, 3), _win_rate(room,seq,5), _win_rate(room,seq,7)
    if not meta.get("use_rate", False):
        w3=w5=w7=0.0

    if last1 is None: p_m1 = 1/8
    else:
        trans={r:0 for r in ROOMS}
        for a,b in pairwise(seq):
            if a==last1: trans[b]+=1
        s=sum(trans.values())+8; p_m1=(trans[room]+1)/s

    parity=float(room%2); mod3=float(room%3); quadrant=float({1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4}[room])
    feats=[float(long_freq), float(short_freq), d_last1, parity, quadrant, mod3, float(w3), float(w5), float(w7), float(p_m1)]
    if len(feats)<FEATURE_DIM: feats += [0.0]*(FEATURE_DIM-len(feats))
    return feats[:FEATURE_DIM]

# --------------------- history merge -----------------------
def _merge_history(recent_10: List[Dict]) -> List[Dict]:
    old=_load_json(HISTORY_FILE, [])
    mp={}
    for d in old:
        if "issue_id" in d and "killed_room_id" in d:
            mp[int(d["issue_id"])]= {"issue_id": int(d["issue_id"]), "killed_room_id": int(d["killed_room_id"])}
    for d in recent_10:
        if "issue_id" in d and "killed_room_id" in d:
            mp[int(d["issue_id"])] = {"issue_id": int(d["issue_id"]), "killed_room_id": int(d["killed_room_id"])}
    merged=sorted(mp.values(), key=lambda x: x["issue_id"], reverse=True)[:MAX_HISTORY]
    merged=list(reversed(merged)); _save_json(HISTORY_FILE, merged)
    return merged

# ------------------ long/short term modules ----------------
def _window(seq: List[int], n: int) -> List[int]:
    return seq[-n:] if len(seq)>=n else seq[:]

def _repeat_prob_given_last(seq: List[int], room: int, lookback: int=30) -> float:
    w=_window(seq, lookback)
    if len(w)<2: return 0.25
    total=0; rep=0
    for a,b in pairwise(w):
        if a==room:
            total+=1
            if b==room: rep+=1
    if total==0: return 0.25
    return (rep+1)/float(total+4)

def _motif_next(last: List[int]) -> Optional[int]:
    if len(last)<3: return None
    a,b,c = last[-3], last[-2], last[-1]
    def norm(d):
        d=((d+8)%8)
        if d>4: d-=8
        return d
    d1=norm(b-a); d2=norm(c-b)
    if d1==d2 and abs(d2) in (1,2,3):
        return ((c + d2 - 1) % 8) + 1
    if c==b==a:
        return c
    if d1==1 and d2==-1: return ((c-1-1)%8)+1
    if d1==-1 and d2==1: return ((c+1-1)%8)+1
    return None

def _markov2_probs(seq: List[int], lookback: int=30) -> Dict[int,float]:
    w=_window(seq, lookback)
    if len(w)<2: return {r: 1/8 for r in ROOMS}
    x1, x2 = w[-2], w[-1]
    cnt={r:1 for r in ROOMS}; tot=8
    for a,b,c in zip(w, w[1:], w[2:]):
        if a==x1 and b==x2:
            cnt[c]+=1; tot+=1
    return {r: cnt[r]/float(tot) for r in ROOMS}

def _short_term_penalties(seq: List[int], ens_probs: Dict[int,float], governor: Dict)->Dict[int,float]:
    if not seq: return {r:0.0 for r in ROOMS}
    cfg=governor.get("tactics", {})
    motif_k=int(cfg.get("motif_k", 6)); band=int(cfg.get("neighbor_band",1))
    max_add=float(cfg.get("max_tactical_pen",0.55))
    last1=seq[-1]
    window=seq[-motif_k:] if len(seq)>=motif_k else seq[:]
    pen={r:0.0 for r in ROOMS}

    nxt=_motif_next(window)
    if nxt is not None:
        pen[nxt]+=0.35
        for r in ROOMS:
            if circ_dist(r, nxt)==1:
                pen[r]+=0.15

    for r in ROOMS: pen[r]=min(max_add, pen[r])
    return pen

# --------------------- state load/save ---------------------
def _ensure_weights_from_meta(meta: Dict)->tuple:
    return float(meta.get("w_lr",0.5)), float(meta.get("w_mlp",0.45)), float(meta.get("w_beta",0.05))

def _load_state():
    d=_load_json(STATE_PRO, None)
    if not d:
        old=_load_json(STATE_OLD, None)
        if not old: return None
        try:
            sc=OnlineScaler.from_dict(old.get("scaler")) if old.get("scaler") else OnlineScaler(FEATURE_DIM)
            lr=OnlineLogRegAdam.from_dict(old.get("logreg")) if old.get("logreg") else OnlineLogRegAdam(FEATURE_DIM)
            mlp=TinyMLP.from_dict(old.get("mlp")) if old.get("mlp") else TinyMLP(FEATURE_DIM,16)
            meta=_ensure_meta(old.get("meta", None))
            gov=_ensure_governor(old.get("governor", {}))
            hedge=old.get("hedge", {"w":[0.5,0.45,0.05]})
            return {"scaler":sc,"logreg":lr,"mlp":mlp,"meta":meta,"governor":gov,"hedge":hedge}
        except Exception:
            return None
    try:
        sc=OnlineScaler.from_dict(d.get("scaler"))
        lr=OnlineLogRegAdam.from_dict(d.get("logreg"))
        mlp=TinyMLP.from_dict(d.get("mlp"))
        meta=_ensure_meta(d.get("meta", None))
        gov=_ensure_governor(d.get("governor", {}))
        hedge=d.get("hedge", {"w":[0.5,0.45,0.05]})
        return {"scaler":sc,"logreg":lr,"mlp":mlp,"meta":meta,"governor":gov,"hedge":hedge}
    except Exception:
        return None

def _save_state(scaler, logreg, mlp, meta, governor, hedge):
    if not scaler or not logreg or not mlp: return
    obj={"scaler": scaler.to_dict(), "logreg": logreg.to_dict(), "mlp": mlp.to_dict(),
         "meta": _ensure_meta(meta), "governor": governor, "hedge": hedge}
    _save_json(STATE_PRO, obj)

# -------------- auto internal outcome update ----------------
def _append_knowledge(record: dict):
    try:
        with open(KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")
    except Exception:
        pass

def _auto_update_last_result(governor: Dict, merged_history: List[Dict]):
    if not merged_history: return
    last_issue=int(merged_history[-1]["issue_id"])
    if governor.get("last_seen_issue")==last_issue:
        return
    killed=int(merged_history[-1]["killed_room_id"])
    prev_choice=governor.get("prev_choice")
    if prev_choice is not None:
        win = (int(prev_choice) != killed)
        if win:
            governor["total_wins"]=int(governor.get("total_wins",0))+1
            governor["win_streak"]=int(governor.get("win_streak",0))+1
            governor["best_streak"]=max(int(governor.get("best_streak",0)), int(governor["win_streak"]))
            governor["loss_streak"]=0
            _append_knowledge({"type":"win","ai_choice":int(prev_choice),"actual":killed,"ts":time.time()})
        else:
            governor["total_losses"]=int(governor.get("total_losses",0))+1
            governor["loss_streak"]=int(governor.get("loss_streak",0))+1
            governor["win_streak"]=0
            ll = governor.get("last_losses", [])
            ll.append({"issue": last_issue, "killed": killed, "ai": int(prev_choice)})
            if len(ll)>4: ll.pop(0)
            governor["last_losses"]=ll
            _append_knowledge({"type":"loss","ai_choice":int(prev_choice),"actual":killed,"ts":time.time()})
    governor["last_seen_issue"]=last_issue

# ----------------------- training step ---------------------
def _train_once(scaler, logreg, mlp, meta, seq_ctx: List[int], killed_room: int, stats_100: Dict[int,int]):
    feats_map={}
    for r in ROOMS:
        f=_features(r, stats_100, seq_ctx, meta)
        scaler.partial_fit(f); xs=scaler.transform(f)
        feats_map[r]=xs
    rep_pos=3
    for r in ROOMS:
        y=1 if r==killed_room else 0
        xs=feats_map[r]
        rep=rep_pos if y==1 else 1
        for _ in range(rep):
            logreg.step(xs, y, lr=meta["lr"], l2=meta["l2"], eps=meta["adam_eps"], clip=meta["grad_clip"])
            mlp.step(xs, y, lr=meta["lr"], l2=meta["l2"], eps=meta["adam_eps"], clip=meta["grad_clip"], dropout_p=meta["dropout"])

# ------------------------- main API ------------------------
def choose_safe_room(stats_100: Dict, recent_10: List[Dict]) -> int:
    stats_100={int(k): int(v) for k,v in stats_100.items()}
    merged=_merge_history(recent_10)
    seq_all=[int(x["killed_room_id"]) for x in merged]
    st=_load_state()
    if st is None:
        scaler=OnlineScaler(FEATURE_DIM); logreg=OnlineLogRegAdam(FEATURE_DIM)
        mlp=TinyMLP(FEATURE_DIM,16); meta=_ensure_meta(None); governor=_ensure_governor({}); hedge={"w":[0.5,0.45,0.05]}
    else:
        scaler,logreg,mlp,meta,governor,hedge = st["scaler"], st["logreg"], st["mlp"], _ensure_meta(st["meta"]), _ensure_governor(st["governor"]), st["hedge"]

    _auto_update_last_result(governor, merged)

    for i in range(len(seq_all)):
        ctx=seq_all[:i]; killed=seq_all[i]
        _train_once(scaler, logreg, mlp, meta, ctx, killed, stats_100)

    seq = seq_all[-10:]
    seq30 = seq_all[-30:]

    feats_map={}
    for r in ROOMS:
        f=_features(r, stats_100, seq, meta)
        if scaler.n==0: scaler.partial_fit(f)
        feats_map[r]=scaler.transform(f)

    w1,w2,w3 = _ensure_weights_from_meta(meta)
    probs_lr = {r: (logreg.predict_proba(feats_map[r]) if logreg.t>0 else 0.12) for r in ROOMS}
    probs_m  = {}
    for r in ROOMS:
        if mlp.t>0: m,_ = mlp.predict_mc(feats_map[r], T=6, dropout_p=meta["dropout"])
        else: m=0.12
        probs_m[r]=m
    probs_b = {r: 1/8 for r in ROOMS}
    ens = {r: max(min(w1*probs_lr[r]+w2*probs_m[r]+w3*probs_b[r], 0.999), 1e-4) for r in ROOMS}

    last1 = seq[-1] if seq else None
    cd = governor.get("cooldown", {r:0 for r in ROOMS})
    for rr in ROOMS: cd[rr]=max(0, int(cd.get(rr,0))-int(governor.get("cooldown_decay",1)))
    # không phạt mặc định phòng vừa xuất hiện & láng giềng
    pen_cd = {r: float(governor.get("cooldown_strength",0.15)) * float(cd.get(r,0)) for r in ROOMS}

    tactical_pen = _short_term_penalties(seq, ens, governor)

    mk2 = _markov2_probs(seq30, lookback=30)
    def _repeat_prob_given_last(seq: List[int], room: int, lookback: int=30) -> float:
        w=_window(seq, lookback)
        if len(w)<2: return 0.25
        total=0; rep=0
        for a,b in pairwise(w):
            if a==room:
                total+=1
                if b==room: rep+=1
        if total==0: return 0.25
        return (rep+1)/float(total+4)
    rep_p = _repeat_prob_given_last(seq30, last1, lookback=30) if last1 is not None else 0.25

    bonus = {r:0.0 for r in ROOMS}
    if last1 is not None and rep_p < 0.20:
        bonus[last1] += 0.35
        nxt = _motif_next(seq[-6:] if len(seq)>=6 else seq[:])
        if nxt is not None:
            for r in ROOMS:
                if circ_dist(r, last1)==1: bonus[r]+=0.20
        else:
            for r in ROOMS:
                if circ_dist(r, last1)==1: bonus[r]+=0.12

    protect_active = int(governor.get("loss_streak",0))>=2
    smart_boost = {r:0.0 for r in ROOMS}
    if protect_active:
        last_losses = governor.get("last_losses", [])[-2:]
        ai_safe_candidates = [int(x["ai"]) for x in last_losses if "ai" in x]
        for r in ai_safe_candidates:
            smart_boost[r] += 0.45
            for q in ROOMS:
                if circ_dist(q, r)==1: smart_boost[q]+=0.15

    final_score={}
    for r in ROOMS:
        s = ens[r] + pen_cd[r] + tactical_pen[r] - bonus[r] - smart_boost[r]
        s += 1e-6*r
        final_score[r]=s

    ordered = sorted(ROOMS, key=lambda k: final_score[k])
    choice = ordered[0]

    eps = meta.get("epsilon", 0.04) if not protect_active else 0.0
    if random.random()<eps and len(ordered)>=2:
        choice = random.choice(ordered[:min(3,len(ordered))])

    cr = governor["chosen_recent"]; cr.append(choice)
    if len(cr)>governor["chosen_max"]: del cr[0]
    if len(cr)>=3 and cr[-1]==cr[-2]==cr[-3]==choice and len(ordered)>=2:
        choice = ordered[1]; cr[-1]=choice

    governor["prev_choice"]=int(choice)
    governor["cooldown"]=cd
    _save_state(scaler, logreg, mlp, meta, governor, hedge)
    return int(choice)

# -------------------- evaluate_outcome (compat) --------------
def evaluate_outcome(ai_choice: Optional[int], actual_killed: Optional[int], governor: dict) -> dict:
    res={"win": None, "loss_streak": int(governor.get("loss_streak",0) or 0)}
    if ai_choice is None or actual_killed is None: return res
    win = (int(ai_choice) != int(actual_killed)); res["win"]=win
    return res

# ------------------------ refresher -------------------------
def start_history_refresher(fetch_recent_func, asset: str="BUILD", interval_sec: int=300):
    stop_history_refresher()
    global _REFRESH_T, _REFRESH_STOP
    _REFRESH_STOP = threading.Event()
    def _worker():
        while not _REFRESH_STOP.is_set():
            try:
                recent = fetch_recent_func()
                if isinstance(recent, list):
                    _merge_history(recent)
            except Exception:
                pass
            for _ in range(int(interval_sec*10)):
                if _REFRESH_STOP.is_set(): break
                time.sleep(0.1)
    t=threading.Thread(target=_worker, name="HistoryRefresherPROppFix", daemon=True)
    t.start(); _REFRESH_T=t
    return True

def stop_history_refresher():
    global _REFRESH_T, _REFRESH_STOP
    if _REFRESH_T and _REFRESH_T.is_alive():
        _REFRESH_STOP.set(); _REFRESH_T.join(timeout=1.0)
    _REFRESH_T=None
    return True