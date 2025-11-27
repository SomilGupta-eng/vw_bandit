import requests
import time
import random

BASE_URL = "http://localhost:8000"

def test_bandit_loop():
    for i in range(20):
        context = {
            "page_type": random.choice(["pdp", "plp", "blog"]),
            "device": random.choice(["desktop", "mobile", "tablet"]),
            "hover": random.choice(["none", "low", "medium", "high"]),
            "scroll": random.choice(["low", "mid", "high"]),
            "time_on_page": random.choice(["very_short", "short", "medium", "long"]),
            "clicked_offer": random.choice([True, False])
        }
        if context["page_type"] == "pdp":
            actions = [
                {"id": "reviews_hero", "type": "reviews", "slot": "hero", "style": "detailed"},
                {"id": "faq_middle", "type": "faq", "slot": "middle", "style": "compact"},
                {"id": "offer_bottom", "type": "offer", "slot": "bottom", "style": "prominent"}
            ]
        else:
            actions = [
                {"id": "reco_hero", "type": "reco", "slot": "hero", "style": "carousel"},
                {"id": "reviews_middle", "type": "reviews", "slot": "middle", "style": "compact"}
            ]

        resp = requests.post(f"{BASE_URL}/decide", json={"context": context, "actions": actions})
        decision = resp.json()
        print(f"Decision {i+1}: {decision['chosen_action']['id']} (prob: {decision['chosen_prob']:.3f})")

        time.sleep(0.5)
        reward = 1.0 if random.random() < 0.3 else 0.0  # Simulate 30% chance of positive engagement
        requests.post(f"{BASE_URL}/reward", json={"decision_id": decision["decision_id"], "reward": reward})
        print(f"  → Reward: {reward}")

if __name__ == "__main__":
    test_bandit_loop()
