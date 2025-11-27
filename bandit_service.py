import random
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import vowpalwabbit
from datetime import datetime

app = FastAPI(title="Contextual Bandit Service")

# VW model instance (cb_explore_adf, 4 actions with DR, epsilon-greedy)
vw = vowpalwabbit.Workspace("--cb_explore_adf --cb_type dr --epsilon 0.1 --quiet")

class Context(BaseModel):
    page_type: str        # e.g. 'pdp', 'plp', 'blog'
    device: str           # 'desktop', 'mobile', 'tablet'
    hover: str            # 'none', 'low', 'medium', 'high'
    scroll: str           # 'low', 'mid', 'high'
    time_on_page: str     # 'very_short', 'short', 'medium', 'long'
    clicked_offer: bool   # True/False

class CandidateAction(BaseModel):
    id: str               # unique string id
    type: str             # e.g. 'reviews', 'faq', 'offer', 'reco'
    slot: str             # e.g. 'hero', 'middle', 'bottom'
    style: str            # e.g. 'compact', 'detailed', 'prominent'

class DecisionRequest(BaseModel):
    context: Context
    actions: List[CandidateAction]

class RewardRequest(BaseModel):
    decision_id: str
    reward: float         # 0.0 to 1.0

# In-memory decision log for demo (use DB in prod)
decision_log = {}

def build_adf_predict_example(ctx: Context, actions: List[CandidateAction]) -> str:
    shared = (
        f"shared |ctx "
        f"page_type={ctx.page_type} "
        f"device={ctx.device} "
        f"hover={ctx.hover} "
        f"scroll={ctx.scroll} "
        f"time_on_page={ctx.time_on_page} "
        f"clicked_offer={str(ctx.clicked_offer).lower()}"
    )
    
    lines = [shared]
    for action in actions:
        line = (
            f"|a type={action.type} slot={action.slot} style={action.style}"
        )
        lines.append(line)
    
    return "\n".join(lines) + "\n"

def build_adf_learn_example(ctx: Context, actions: List[CandidateAction], chosen_idx:int, cost:float, prob:float) -> str:
    shared = (
        f"shared |ctx "
        f"page_type={ctx.page_type} "
        f"device={ctx.device} "
        f"hover={ctx.hover} "
        f"scroll={ctx.scroll} "
        f"time_on_page={ctx.time_on_page} "
        f"clicked_offer={str(ctx.clicked_offer).lower()}"
    )
    
    lines = [shared]
    for i, action in enumerate(actions):
        label_prefix = f"0:{cost}:{prob} " if i == chosen_idx else ""
        line = f"{label_prefix}|a type={action.type} slot={action.slot} style={action.style}"
        lines.append(line)
    return "\n".join(lines) + "\n"

@app.post("/decide")
async def decide(request: DecisionRequest):
    global decision_log
    ex = build_adf_predict_example(request.context, request.actions)
    scores = vw.predict(ex)

    # Normalize scores if they don't sum to 1
    total = sum(scores)
    probs = [s / total for s in scores]

    r = random.random()
    cum = 0.0
    chosen_idx = 0
    chosen_prob = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            chosen_idx = i
            chosen_prob = p
            break

    chosen_action = request.actions[chosen_idx]
    decision_id = f"{datetime.now().isoformat()}_{random.randint(1000,9999)}"
    
    decision_log[decision_id] = {
        "context": request.context.dict(),
        "actions": [a.dict() for a in request.actions],
        "chosen_idx": chosen_idx,
        "chosen_prob": chosen_prob,
        "timestamp": datetime.now().isoformat()
    }

    return {
        "decision_id": decision_id,
        "chosen_action": chosen_action.dict(),
        "chosen_prob": chosen_prob,
        "all_probs": list(zip(range(len(probs)), probs))
    }

@app.post("/reward")
async def reward(req: RewardRequest):
    global decision_log, vw
    if req.decision_id not in decision_log:
        raise HTTPException(status_code=404, detail="Decision not found")
    decision = decision_log[req.decision_id]
    ctx = Context(**decision["context"])
    actions = [CandidateAction(**a) for a in decision["actions"]]
    chosen_idx = decision["chosen_idx"]
    chosen_prob = decision["chosen_prob"]

    cost = 1.0 - req.reward
    learn_ex = build_adf_learn_example(ctx, actions, chosen_idx, cost, chosen_prob)
    vw.learn(learn_ex)
    vw.save("cb_model.vw")

    return {"status": "learned", "reward": req.reward}

@app.get("/status")
async def status():
    return {
        "model_loaded": vw is not None,
        "decisions_logged": len(decision_log),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
