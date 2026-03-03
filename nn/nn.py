import sys
import json
import os
import pickle
import numpy as np
import hashlib
import logging
import re
import uuid
import time
import traceback
import pandas as pd
import shutil

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode, coalesce
from pyspark.sql.types import *

from notebookutils import mssparkutils
from sklearn.neural_network import MLPClassifier

# --- LOGGING SETUP ---
logger = logging.getLogger("NN")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def log(msg):
    logger.info(msg)

# --- CONFIGURATION ---
# Event Hub connection strings are read from environment variables to avoid
# committing secrets to source control.
INPUT_CONN_STR = os.environ.get("NN_INPUT_CONN_STR", "")

# *** NEW: OUTPUT STREAM CONFIGURATION ***
OUTPUT_CONN_STR = os.environ.get("NN_OUTPUT_CONN_STR", "")

# PATHS
LOCAL_DIR = "/tmp/NN"
TARGET_DIR = "Files/NN"
TARGET_FILE = "Files/NN/neural_brain.pkl"

ACTIONS = ["faq-component", "chat-widget", "customer-reviews"]
N_FEATURES = 100 
EPSILON = 0.1 

# State Tracking
active_suggestions = {} 
user_history = {} 

# --- REWARDS ---
WEIGHTS = { 
    'blur': -5.0, 
    'hidden': -2.0,      
    'scroll_depth': 0.5, 
    'hover_sec': 0.1,    
    'click': 2.0,       
    'conversion': 20.0  
}

# --- HELPERS ---
def safe_get(obj, key, default=None):
    if obj is None: return default
    if isinstance(obj, dict): return obj.get(key, default)
    if hasattr(obj, key): return getattr(obj, key, default)
    try: return obj[key]
    except: return default

def clean_text(text):
    if not text: return []
    words = re.findall(r'\w+', str(text).lower())
    return [w for w in words if len(w) > 3] 

def detect_conversion_intent(text):
    if not text: return False
    t = str(text).lower()
    intent_words = ['checkout', 'buy', 'purchase', 'add to cart', 'place order', 'payment', 'apply']
    return any(w in t for w in intent_words)

# --- BRAIN ---
class NNAgent:
    def __init__(self, actions, n_features):
        self.actions = actions
        self.n_features = n_features
        self.models = []
        for _ in range(len(actions)):
            model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', warm_start=False)
            X_init = np.zeros((2, n_features))
            y_init = [0, 1] 
            model.partial_fit(X_init, y_init, classes=[0, 1])
            self.models.append(model)

    def predict(self, context_vec):
        scores = []
        for model in self.models:
            try:
                prob = model.predict_proba([context_vec])[0][1]
                scores.append(prob)
            except:
                scores.append(0.5)
        return scores

    def train(self, context_vec, arm_idx, reward):
        y = [1] if reward > 0 else [0]
        self.models[arm_idx].partial_fit([context_vec], y, classes=[0, 1])

# --- FEATURE EXTRACTION ---
def get_feature_vector(row):
    events = row.get('events', [])
    if isinstance(events, np.ndarray): events = events.tolist()
    
    # 1. Default assumes Mobile
    ctx = { "dev": "mobile", "source": "direct", "intent": "browse", "focus": "body" }
    
    swipe_count = 0
    max_scroll = 0.0
    hover_reading = 0
    text_tokens = set()
    
    # Debug vars
    found_ua = "None"

    if events:
        # --- FIX: Scan ALL events to find a valid UserAgent ---
        ua = ""
        url = ""
        for e in events:
            current_ua = str(safe_get(e, 'userAgent') or "").lower()
            current_url = str(safe_get(e, 'url') or "").lower()
            
            if current_ua: ua = current_ua
            if current_url: url = current_url
            if ua and url: break 
        
        # --- LOGIC: Set Device Type ---
        if ua:
            found_ua = ua  # Store for logging
            # Only switch to desktop if we are SURE it's not mobile
            if not ('android' in ua or 'mobile' in ua or 'iphone' in ua):
                ctx['dev'] = 'desktop'
        
        if 'utm_source' in url or 'facebook' in ua: ctx['source'] = 'ads'
        if '/cart' in url: ctx['intent'] = 'checkout'
        
        # --- NEW LOG: Shows User ID + Detected Device + The UA string it found ---
        sid_short = str(row.get('session_id', 'unknown'))[-4:]
        log(f"[DEVICE] User: {sid_short} | Detected: {ctx['dev'].upper()} | UA: {found_ua[:60]}...") 

        # Metric extraction
        for e in events:
            e_type = safe_get(e, 'type')
            data = safe_get(e, 'data', {})
            
            if e_type == 'swipe': swipe_count += 1
            if e_type == 'scroll': 
                d = safe_get(data, 'scrollDepth', 0)
                if d: max_scroll = max(max_scroll, float(d))
            
            if e_type in ['hover', 'click', 'touch_start', 'button_click']:
                el = safe_get(data, 'element', {})
                tgt = safe_get(data, 'target', {})
                t1 = safe_get(el, 'textContent')
                t2 = safe_get(tgt, 'textContent')
                if t1: text_tokens.update(clean_text(t1))
                if t2: text_tokens.update(clean_text(t2))
                
                dur = safe_get(data, 'duration', 0)
                if dur and dur > 1000: hover_reading += 1

    features = []
    features.append(f"dev={ctx['dev']}")
    features.append(f"src={ctx['source']}")
    features.append(f"intent={ctx['intent']}")
    
    if ctx['dev'] == 'mobile': features.append(f"swipes={int(swipe_count/5)}")
    else: features.append(f"reads={hover_reading}")
    
    for w in list(text_tokens)[:15]: features.append(f"word={w}")
    
    vec = np.zeros(N_FEATURES)
    for f in features:
        h = int(hashlib.md5(f.encode()).hexdigest(), 16)
        vec[h % N_FEATURES] = 1.0 
    
    return vec, ctx

# --- FEEDBACK CHECKER ---
def calculate_feedback(events, suggested_id):
    if not events: return False, 0.0
    
    score = 0.0
    direct_hit = False
    has_exit = False
    
    timestamps = [safe_get(e, 'timestamp', 0) for e in events]
    valid_ts = [t for t in timestamps if t > 0]
    duration = 1.0
    if valid_ts and len(valid_ts) > 1:
        duration = (max(valid_ts) - min(valid_ts)) / 1000.0
    if duration < 1: duration = 1.0

    for e in events:
        e_type = safe_get(e, 'type')
        data = safe_get(e, 'data', {})
        
        tgt = safe_get(data, 'target', {})
        el = safe_get(data, 'element', {})
        tgt_id = safe_get(tgt, 'id')
        el_id = safe_get(el, 'id')
        
        if suggested_id and (tgt_id == suggested_id or el_id == suggested_id):
            direct_hit = True
            
        interaction_text = ""
        if e_type in ['click', 'button_click', 'touch_start', 'touch_end']:
            interaction_text += str(safe_get(tgt, 'textContent') or "")
            interaction_text += str(safe_get(el, 'textContent') or "")
            
            if detect_conversion_intent(interaction_text):
                score += WEIGHTS['conversion'] 
            else:
                score += WEIGHTS['click']

        if e_type == 'scroll' or e_type == 'swipe': 
            d = safe_get(data, 'scrollDepth', 0)
            if d: score += (float(d) * WEIGHTS['scroll_depth'])
            else: score += 0.1 
            
        if e_type in ['window_blur', 'page_unload', 'session_end']:
            has_exit = True

    velocity = score / duration
    
    if direct_hit: return 'positive', velocity
    if has_exit: return 'negative', velocity
    return 'neutral', velocity

# --- METRIC LOGGING ---
def log_metrics_to_lakehouse(batch_id, events_list):
    """
    Saves performance data to a Delta Table for analysis.
    """
    metrics_data = []
    
    for evt in events_list:
        metrics_data.append({
            "batch_id": batch_id,
            "time": time.time(),
            "session_id": evt['sid'],
            "device": evt['device_name'],
            "action_chosen": evt['action'],
            "model_confidence": float(evt['score']),
            "reward_received": float(evt['reward'])
        })
    
    if not metrics_data: return

    # Create DataFrame
    df_metrics = pd.DataFrame(metrics_data)
    spark_metrics = spark.createDataFrame(df_metrics)
    
    # Write to Delta Table (Append Mode)
    METRICS_TABLE = "Tables/NN_performance"
    # NEW (Table based - forces registration)
    spark_metrics.write.format("delta").mode("append").saveAsTable("NN_performance")
    log(f"METRICS: Saved {len(metrics_data)} rows to {METRICS_TABLE}")


# --- BATCH PROCESSOR ---
def process_batch(df_batch, batch_id):
    if not os.path.exists(LOCAL_DIR):
        try: os.makedirs(LOCAL_DIR, exist_ok=True)
        except: pass
    
    # --- LOAD BRAIN (WITH CORRUPTION HANDLING) ---
    agent = None
    try:
        if mssparkutils.fs.exists(TARGET_FILE):
            # Use unique temp file to avoid read locks
            load_path = f"{LOCAL_DIR}/load_{uuid.uuid4().hex}.pkl"
            mssparkutils.fs.cp(TARGET_FILE, f"file://{load_path}", True)
            
            try:
                with open(load_path, 'rb') as f: agent = pickle.load(f)
                
                # Check 1: Dimensions
                if not hasattr(agent, 'n_features') or agent.n_features != N_FEATURES:
                    log("WARNING: Brain mismatch. Resetting.")
                    agent = None
                    
            except EOFError:
                # Check 2: File corruption (0 bytes)
                log("WARNING: Saved brain is corrupted/empty. Deleting and restarting.")
                mssparkutils.fs.rm(TARGET_FILE, True) # Delete bad file from Lakehouse
                agent = None
                
            finally:
                if os.path.exists(load_path): os.remove(load_path)
        else:
            log("STATUS: No existing brain found in OneLake.")
            
    except Exception as e:
        log(f"Load Warning: {e}")
        
    if agent is None:
        agent = NNAgent(ACTIONS, N_FEATURES)
        log("STATUS: Started FRESH brain.")

    # 1. CONVERT
    pdf = df_batch.toPandas()
    if pdf.empty: return

    # 2. GROUP BY USER
    pdf['events'] = pdf['events'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else []))
    
    grouped_pdf = pdf.groupby('session_id').agg({
        'events': 'sum', 
        'shown_component': 'last'
    }).reset_index()

    log(f"--- Batch {batch_id}: Processing {len(grouped_pdf)} unique users ---")
    updated = False

    # NEW: List to hold metrics for this batch
    batch_metrics = []
    
    # NEW: List to hold outgoing suggestions for the output stream
    new_suggestions_to_stream = []

    for index, row in grouped_pdf.iterrows():
        try:
            sid = row['session_id']
            if not sid: continue

            ctx_vec, ctx_dict = get_feature_vector(row)
            events = row['events']

            # --- DECISION PHASE ---
            if sid not in active_suggestions:
                _, pre_velocity = calculate_feedback(events, None)
                scores = agent.predict(ctx_vec)
                score_map = {ACTIONS[i]: s for i, s in enumerate(scores)}
                
                formatted_scores = ", ".join([f"{k}={v:.4f}" for k,v in score_map.items()])
                log(f"User {sid[-4:]}: [SCORES] {formatted_scores}")
                
                seen = user_history.get(sid, set())
                candidates = {k: v for k, v in score_map.items() if k not in seen}
                
                if not candidates: continue 
                
                # --- NEW: Epsilon-Greedy Logic ---
                # 1. Decide: Explore (Random) vs Exploit (Best Score)
                is_exploring = False
                candidate_keys = list(candidates.keys())
                
                if np.random.random() < EPSILON:
                    # Explore: Pick a random available action
                    # Using numpy to pick a random index
                    rand_idx = np.random.randint(len(candidate_keys))
                    rec = candidate_keys[rand_idx]
                    is_exploring = True
                    log(f"User {sid[-4:]}: [EXPLORE] Randomly picked {rec}")
                else:
                    # Exploit: Pick the action with the highest score
                    rec = max(candidates, key=candidates.get)
                    is_exploring = False

                # 2. Capture the model's confidence for this action (Even if random!)
                chosen_score = candidates[rec]
                
                inj_id = f"inj_{str(uuid.uuid4())[:8]}"
                
                # 3. Save state (INCLUDING SCORE)
                active_suggestions[sid] = {
                    'action': rec, 
                    'id': inj_id,
                    'pre_velocity': pre_velocity, 
                    'score': chosen_score, # <--- Fixes the "0.0" bug
                    'is_exploring': is_exploring,
                    'ts': time.time()
                }
                
                if sid not in user_history: user_history[sid] = set()
                user_history[sid].add(rec)
                
                log(f"User {sid[-4:]}: [SUGGEST] {rec} (ID: {inj_id})")
                
                # --- NEW: ADD TO OUTPUT STREAM LIST ---
                new_suggestions_to_stream.append({
                    "session_id": sid,
                    "component": rec,
                    "injection_id": inj_id,
                    "timestamp": time.time()
                })
                
                updated = True # <--- CRITICAL FIX: Save state even if only suggesting
                
            # --- FEEDBACK PHASE ---
            else:
                history = active_suggestions[sid]
                inj_id = history['id']
                pre_velocity = history['pre_velocity']
                action_idx = ACTIONS.index(history['action'])
                
                outcome, post_velocity = calculate_feedback(events, inj_id)
                delta = post_velocity - pre_velocity
                
                reward = 0.0
                trained = False
                
                if outcome == 'positive':
                    log(f"User {sid[-4:]}: [SUCCESS] Direct Hit on {inj_id} (+1.0)")
                    reward = 1.0; trained = True
                    
                elif outcome == 'negative':
                    log(f"User {sid[-4:]}: [FAIL] User left without interacting (-0.5)")
                    reward = -0.5; trained = True
                    
                elif outcome == 'neutral':
                    if delta > 0.5:
                        log(f"User {sid[-4:]}: [PARTIAL] Engagement Spike (+{delta:.2f}) -> (+0.5)")
                        reward = 0.5; trained = True
                
                if trained:
                    agent.train(ctx_vec, action_idx, reward)
                    updated = True
                    del active_suggestions[sid]

                    batch_metrics.append({
                        'sid': sid,
                        'ctx': ctx_vec,
                        'device_name': ctx_dict['dev'],
                        'action': history['action'],
                        'score': history.get('score', 0.5),
                        'reward': reward
                    })

        except Exception as e:
            traceback.print_exc()

    # --- SAVE METRICS TO DELTA TABLE IF ANY ---
    if batch_metrics:
        try:
            log_metrics_to_lakehouse(batch_id, batch_metrics)
        except Exception as e:
            log(f"Metric Save Failed: {e}")

    # --- NEW: SEND SUGGESTIONS TO OUTPUT EVENT STREAM ---
    if new_suggestions_to_stream:
        try:
            # 1. Create rows containing a JSON "body"
            out_rows = [{"body": json.dumps(item)} for item in new_suggestions_to_stream]
            df_out = spark.createDataFrame(out_rows)
            
            # 2. Configure output write
            # We encrypt the connection string here using the global Spark Context (sc)
            eh_out_conf = {
                "eventhubs.connectionString": sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(OUTPUT_CONN_STR)
            }
            
            # 3. Write to Event Hub
            df_out.write.format("eventhubs").options(**eh_out_conf).save()
            log(f"STREAM OUT: Sent {len(new_suggestions_to_stream)} suggestions to Output Hub.")
            
        except Exception as e:
            log(f"STREAM OUT ERROR: {e}")

    # --- 10. SAVE LOGIC (UNIQUE FILE SAFE) ---
    if updated:
        try:
            # 1. Create a UNIQUE local filename
            unique_name = f"brain_{uuid.uuid4().hex}.pkl"
            save_path = f"{LOCAL_DIR}/{unique_name}"
            
            # 2. Write to Local
            with open(save_path, 'wb') as f:
                pickle.dump(agent, f)
                f.flush()
                os.fsync(f.fileno())
            
            # 3. Upload to OneLake (Overwriting target safely)
            mssparkutils.fs.mkdirs(TARGET_DIR)
            mssparkutils.fs.cp(f"file://{save_path}", TARGET_FILE, True)
            
            # 4. Cleanup
            os.remove(save_path)
            
            log(f"CONFIRMATION: Brain uploaded to {TARGET_FILE}")
            
        except Exception as e:
            log(f"CRITICAL SAVE FAILURE: {e}")

# --- START ---
spark = SparkSession.builder.appName("TextAwareNN").getOrCreate()
sc = spark.sparkContext
read_conf = {"eventhubs.connectionString": sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(INPUT_CONN_STR)}
df_raw = spark.readStream.format("eventhubs").options(**read_conf).load()

# --- SCHEMA ---
nested_schema = ArrayType(StructType([
    StructField("payload", StructType([
        StructField("sessionId", StringType(), True),
        StructField("shownComponent", StringType(), True), 
        StructField("metadata", StructType([
            StructField("sessionId", StringType(), True)
        ])),
        StructField("events", ArrayType(StructType([
            StructField("type", StringType(), True),
            StructField("timestamp", LongType(), True),
            StructField("url", StringType(), True),
            StructField("userAgent", StringType(), True),
            StructField("data", StructType([
                StructField("scrollY", LongType(), True),
                StructField("scrollDepth", DoubleType(), True),
                StructField("duration", LongType(), True),
                StructField("timeSpent", LongType(), True),
                StructField("hidden", BooleanType(), True),
                StructField("textContent", StringType(), True),
                StructField("x", LongType(), True), 
                StructField("y", LongType(), True),
                StructField("referrer", StringType(), True), 
                StructField("distance", DoubleType(), True),
                StructField("element", StructType([
                    StructField("textContent", StringType(), True),
                    StructField("id", StringType(), True),
                    StructField("boundingRect", StructType([
                        StructField("y", DoubleType(), True)
                    ]))
                ])),
                StructField("target", StructType([
                    StructField("textContent", StringType(), True),
                    StructField("id", StringType(), True)
                ]))
            ]))
        ])))
    ]))
]))

df_parsed = df_raw.withColumn("body_str", col("body").cast("string")) \
    .select(from_json("body_str", nested_schema).alias("data_array")) \
    .select(explode("data_array").alias("data_struct")) \
    .select(
        coalesce(col("data_struct.payload.sessionId"), col("data_struct.payload.metadata.sessionId")).alias("session_id"),
        col("data_struct.payload.shownComponent").alias("shown_component"), 
        col("data_struct.payload.events").alias("events")
    )

log("Starting Text-Aware NN (MEMORY FIX)...")
query = df_parsed.writeStream.foreachBatch(process_batch).start()
query.awaitTermination()