import sys
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# --- LOGGING SETUP ---
logger = logging.getLogger("RawLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def log(msg):
    logger.info(msg)

# --- CONFIGURATION ---
# Event Hub connection string is now read from an environment variable to avoid
# committing secrets to source control.
EVENT_HUB_CONN_STR = os.environ.get("EVENT_HUB_CONN_STR", "")

def process_batch(df, batch_id):
    count = df.count()
    log(f"--- Batch {batch_id}: Received {count} events ---")
    
    if count > 0:
        # Collect the first 10 rows to inspect raw structure
        rows = df.limit(10).collect()
        for i, row in enumerate(rows):
            log(f"\n[Event {i+1}]")
            # Log the custom properties (where userId might be hidden)
            log(f"Properties: {row['properties']}")
            # Log the body content
            log(f"Body (String): {row['body_str']}")

# --- MAIN ---
if __name__ == "__main__":
    spark = SparkSession.builder.appName("RawEventHubLogger").getOrCreate()
    sc = spark.sparkContext

    # Setup Event Hubs configuration
    # Note: Using the Spark EventHubs connector utils to encrypt the string
    eh_conf = {
        "eventhubs.connectionString": sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(EVENT_HUB_CONN_STR)
    }

    log("Starting Raw Event Hub Logger...")
    # Extract entity path for logging confirmation
    entity_path = EVENT_HUB_CONN_STR.split("EntityPath=")[-1] if "EntityPath=" in EVENT_HUB_CONN_STR else "Unknown"
    log(f"Listening to Entity: {entity_path}")

    # Read Stream
    # We use 'load()' without extra filtering to get everything
    df_raw = spark.readStream.format("eventhubs").options(**eh_conf).load()

    # Prepare DataFrame: Cast body to string for readability, keep properties
    df_debug = df_raw.withColumn("body_str", col("body").cast("string")) \
                     .select("body_str", "properties")

    # Write to console (stdout) via foreachBatch to view logs in the driver
    query = df_debug.writeStream.foreachBatch(process_batch).start()
    query.awaitTermination()
