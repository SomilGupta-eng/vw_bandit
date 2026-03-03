import time
import json
import uuid
import random
import os
from azure.eventhub import EventHubProducerClient, EventData

# --- 1. CONFIGURATION ---
# Connection details are provided via environment variables to avoid
# committing secrets to source control.
CONNECTION_STR = os.environ.get("TEST_EVENTHUB_CONN_STR", "")
EVENTHUB_NAME = os.environ.get("TEST_EVENTHUB_NAME", "")

def get_fake_session_id():
    # Matches format: "session_wzl0sory2_1765189878174"
    rand_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=9))
    ts = int(time.time() * 1000)
    return f"session_{rand_str}_{ts}"

def generate_browser_context(session_id, user_ip="45.224.51.150"):
    """Generates the realistic 'Linux Headless Bot' signature"""
    now_ms = int(time.time() * 1000)
    return {
        "sessionId": session_id,
        "startTime": now_ms,
        "userAgent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.114 Safari/537.36",
        "language": "en-US",
        "timezone": "Asia/Shanghai",
        "screen": {"width": 1280, "height": 1200, "availWidth": 1280, "availHeight": 1200, "colorDepth": 24, "pixelDepth": 24},
        "viewport": {"width": 1280, "height": 1200, "devicePixelRatio": 1},
        "device": {"type": "desktop", "platform": "Linux x86_64", "maxTouchPoints": 0, "hardwareConcurrency": 64, "deviceMemory": 0, "gpu": "WebKit WebGL"},
        "browser": {"name": "Chrome", "version": "126.0.6478.114", "cookieEnabled": True, "onLine": True, "javaEnabled": False},
        "os": {"name": "Linux", "platform": "Linux x86_64"},
        "connection": {"effectiveType": "4g", "downlink": 10, "rtt": 0, "saveData": False},
        "referrer": "",
        "initialUrl": "http://wanderingbean.in/",
        "title": "BESPOKE HAND CRAFTED COFFEE – The Wandering Bean",
        "userIP": user_ip,
        "fbp": f"fb.1.{now_ms}.{random.randint(10000000000000000, 99999999999999999)}",
        "fbc": None
    }

def send_wrapped_data(client, inner_payload):
    """
    CRITICAL FIX: Wraps the data in a LIST with a 'payload' key 
    to match the schema in your nn.py file.
    Structure: [ { "payload": { ... } } ]
    """
    wrapped_message = [
        { "payload": inner_payload }
    ]
    
    json_str = json.dumps(wrapped_message)
    event_data_batch = client.create_batch()
    event_data_batch.add(EventData(json_str))
    client.send_batch(event_data_batch)
    print(f"-> Sent WRAPPED payload ({len(inner_payload.get('events', []))} events)")

def main():
    try:
        client = EventHubProducerClient.from_connection_string(CONNECTION_STR, eventhub_name=EVENTHUB_NAME)
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    session_id = get_fake_session_id()
    print(f"--- STARTING SIMULATION: {session_id} ---")

    # ==========================================
    # PHASE 1: Session Start + Context
    # ==========================================
    print("\n[PHASE 1] Sending Context...")
    
    browser_data = generate_browser_context(session_id)
    
    # The inner content (matches your 'payload' schema fields)
    inner_payload_start = {
        "sessionId": session_id,
        "shownComponent": None,
        "metadata": {"sessionId": session_id},
        "events": [
            {
                "type": "session_start",
                "timestamp": int(time.time() * 1000),
                "url": "http://wanderingbean.in/",
                "userAgent": browser_data['userAgent'],
                # Detailed data goes here
                "data": {
                    # We merge browser data into 'data' to ensure no field is missing
                    **browser_data,
                    "textContent": "Home Page",
                    "element": {"textContent": "Body", "id": "body-1"},
                    "target": {"textContent": "Window", "id": "window-1"}
                }
            },
            {
                "type": "scroll",
                "timestamp": int(time.time() * 1000) + 200,
                "data": {
                    "scrollY": 100, 
                    "scrollDepth": 0.05,
                    "textContent": None,
                    "target": {"textContent": "Main Wrapper", "id": "wrapper-main"}, 
                    "element": {"textContent": "Body", "id": "body-1"}
                }
            }
        ]
    }
    
    send_wrapped_data(client, inner_payload_start)

    print("\n>>> CHECK SPARK LOGS NOW <<<")
    print(f"Look for: 'User {session_id[-4:]}: Suggested ... with ID 'inj_xxxx'")
    injected_id = input("Paste the Suggested ID here: ").strip()

    # ==========================================
    # PHASE 2: Feedback / Reaction
    # ==========================================
    print(f"\n[PHASE 2] Simulating Reaction to {injected_id}...")
    choice = input("Outcome? (1=Click, 2=Ignore): ")

    events = []
    ts = int(time.time() * 1000) + 5000

    if choice == "1":
        # SUCCESS: Click on the suggested ID
        events.append({
            "type": "click",
            "timestamp": ts,
            "data": {
                "target": {"id": injected_id, "textContent": "Chat Button"}, # <--- MATCH
                "element": {"id": "chat-wrapper", "textContent": "Chat"},
                "textContent": "Chat"
            }
        })
        events.append({
            "type": "visibility_change",
            "timestamp": ts + 1000,
            "data": {"hidden": False, "timeSpent": 20000, "textContent": None}
        })
    else:
        # FAIL: Ignore
        events.append({
            "type": "window_blur",
            "timestamp": ts,
            "data": {"textContent": None}
        })

    inner_payload_feedback = {
        "sessionId": session_id,
        "shownComponent": "chat-widget",
        "metadata": {"sessionId": session_id},
        "events": events
    }

    send_wrapped_data(client, inner_payload_feedback)
    print("\n[DONE] Feedback sent.")
    client.close()

if __name__ == "__main__":
    main()