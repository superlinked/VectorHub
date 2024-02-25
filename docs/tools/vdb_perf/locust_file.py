import threading
import time
import logging
import yaml
from pinecone import Pinecone
import json 
import os

# Env variables
input_file_path = os.environ.get("VECTOR_FILE_PATH")
api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
vector_file_path = input_file_path
loaded_vectors = []
with open(vector_file_path, 'r') as file:
    loaded_vectors = json.load(file)
    for vector in loaded_vectors:
        vector['values'] = [float(value) for value in vector['values']]

# Load the runbook
with open('final_runbook.yaml', 'r') as file:
    runbook = yaml.safe_load(file)

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

def operation_with_timing(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    time_taken = end_time - start_time
    return result, time_taken

def search_operation(vector, index, namespace="ns1"):
    try:
        # Implement search logic based on runbook
        response = index.query(
            namespace=namespace,
            vector=vector,
            top_k=100,
            # include_values=True,
            # include_metadata=True,
        )
        print(convert_response(response=response))
        return response
        # print(response)
        response
        # Log the response or perform other operations
    except Exception as e:
        logger.error(f"Error in search_operation: {e}")

def convert_response(response, start_time, end_time, id):
    # Placeholder values for id, start, and elapsed
    example_id = id
    start_time = start_time
    elapsed_time = end_time

    # Extracting IDs from the response
    response_ids = [int(match['id']) for match in response['matches']]

    # Constructing the output format
    output = {
        "id": example_id,
        "start": start_time,
        "elapsed": elapsed_time,
        "response": response_ids
    }

    return output



def insert_operation(vectors, index, namespace="ns1"):
    for vector in vectors:    
        try:
            index.upsert(
                vectors=[
                    {"id": str(vector["id"]), "values": vector["values"]},
                ],
                namespace="ns1"
            )

        except Exception as e:
            logger.error(f"Error in insert_operation: {e}")

def operation_with_timing(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    time_taken = end_time - start_time
    return result, time_taken

output_file = ""

def threaded_function(target, vectors, thread_count, output_file):
    threads = []
    results = []
    vectors_per_thread = len(vectors) // thread_count
    output_file = output_file
    # Create and start threads
    for i in range(thread_count):
        start_index = i * vectors_per_thread
        end_index = None if i == thread_count - 1 else (i + 1) * vectors_per_thread
        thread_vectors = vectors[start_index:end_index]
        thread = threading.Thread(target=lambda q, vec: q.append(operation_with_timing(target, vec, index)),
                                  args=(results, thread_vectors))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Save results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Usage
num_threads = 1  # Change this to increase or decrease the number of threads

for vector in loaded_vectors[:1]:
    search_operation(vector=vector["values"], index=index)
