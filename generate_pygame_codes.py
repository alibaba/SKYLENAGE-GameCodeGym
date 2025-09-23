from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import jsonlines
import argparse
import tqdm
import traceback
from openai import OpenAI

def ensure_directory_exists(path, type="file"):
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path = Path(path)
    if type == "file":
        parent_dir = os.path.dirname(path)
        os.makedirs(parent_dir, exist_ok=True)
    elif type == "dir":
        os.makedirs(path, exist_ok=True)

def read_jsonl_file(file_name, max_sentence=None):
    data = []
    with jsonlines.open(file_name, "r") as r:
        for i, obj in tqdm.tqdm(enumerate(r)):
            if max_sentence is not None and i >= max_sentence:
                return data
            data.append(obj)
    return data

def write_jsonl_file(objs, path, chunk_size=1, format="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, format, flush=True) as w:
        for i in tqdm.tqdm(range(0, len(objs), chunk_size)):
            w.write_all(objs[i: i + chunk_size])
    print(f"Successfully saving to {path}: {len(objs)}")

def load_config(config_path):
    """Load the configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def init_llm_client(config):
    """Initialize the LLM client"""
    client_config = config.get("client_config", {})
    return OpenAI(
        api_key=client_config.get("api_key", "sk-abc123"),
        base_url=client_config.get("base_url"),
        timeout=client_config.get("timeout", 7200),
        max_retries=client_config.get("max_retries", 10)
    )

def generate_code_from_requirement(client, config, requirement, model_name=None):
    """Generate game code based on the requirement"""
    chat_config = config.get("chat_config", {})
    if model_name:
        chat_config = chat_config.copy()
        chat_config["model"] = model_name
    
    prompt = f"""Generate a complete pygame code based on the following game requirement:

Requirement: {requirement}

Requirements:
1. Generate a complete and runnable pygame code
2. The game should automatically run for 10 seconds and then exit (no manual operation required)
3. Include all necessary import statements, especially `import time`
4. Use pygame and Python standard libraries
5. Ensure clear visual effects for easy observation
6. Add time-based automatic exit mechanism:
   ```python
   import time
   start_time = time.time()
   # In main loop:
   current_time = time.time()
   if current_time - start_time >= 10:  # Run for 10 seconds
       running = False
   ```
7. Add a visual timer showing elapsed time (e.g., "Time: 5.2/10.0s")
8. Set reasonable FPS (30 or 60)

Please output the complete Python code directly.
"""
    
    try:
        response = client.chat.completions.create(
            model=chat_config.get("model"),
            messages=[
                {"role": "system", "content": "You are a pygame game development expert, good at quickly developing small games based on requirements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating code: {e}")
        return None

def extract_code_from_response(response_text):
    """Extract the code from the response"""
    import re
    code_pattern = r'```python\n(.*?)\n```'
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    if matches:
        return matches[0]
    return response_text

def generate_code_worker(task_args):
    """Process a single code generation task"""
    dataset_item = task_args.get("dataset_item", {})
    config = task_args.get("config", {})
    test_model = task_args.get("test_model", None)
    
    game_id = dataset_item.get("id", "unknown")
    requirement = dataset_item.get("requirement", "")
    
    try:
        # Initialize the client
        client = init_llm_client(config)
        
        # Generate code
        generated_response = generate_code_from_requirement(client, config, requirement, test_model)
        if not generated_response:
            return {
                "success": False,
                "game_id": game_id,
                "error": "Failed to generate code"
            }
        
        generated_code = extract_code_from_response(generated_response)
        
        # Build the result
        result = {
            "success": True,
            "game_id": game_id,
            "requirement": requirement,
            "generated_code": generated_code,
            "test_model": test_model,
            "reference_code": dataset_item.get("code", "")  # Keep the reference code
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "game_id": game_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, 
                       default="gamegym_testset/gamegym_testset.jsonl")
    parser.add_argument("--output_path", type=str, 
                       default="evaluation_results/generated_game_codes/")
    parser.add_argument("--config_path", type=str,
                       default="config/config.json")
    parser.add_argument("--test_model", type=str, required=True, help="The name of the model to test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Continue from the last interrupted point")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Load the configuration
    config = load_config(args.config_path)
    print(f"Generating codes with model: {args.test_model}")
    
    # Read the dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = read_jsonl_file(args.dataset_path, max_sentence=args.max_samples)
    
    if args.start_from > 0:
        dataset = dataset[args.start_from:]
    
    print(f"Generating codes for {len(dataset)} games")
    
    # Create the output directory
    output_dir = os.path.join(args.output_path, args.test_model.replace("/", "_"))
    ensure_directory_exists(output_dir, type="dir")
    
    # Output file path
    results_path = os.path.join(output_dir, "generated_codes.jsonl")
    
    # If resume mode, load the existing results
    existing_ids = set()
    if args.resume and os.path.exists(results_path):
        existing_results = read_jsonl_file(results_path)
        existing_ids = {r["game_id"] for r in existing_results if r.get("success", False)}
        print(f"Resume mode: found {len(existing_ids)} existing results")
    
    # Build the task queue (skip the completed ones)
    task_queue = []
    for item in dataset:
        if item.get("id") not in existing_ids:
            task_queue.append({
                "dataset_item": item,
                "config": config,
                "test_model": args.test_model,
            })
    
    print(f"Tasks to process: {len(task_queue)}")
    
    if len(task_queue) == 0:
        print("All tasks already completed!")
        return
    
    # Process the tasks
    task_bar = tqdm.tqdm(total=len(task_queue), desc=f"Generating with {args.workers} workers")
    results = []
    
    # Statistics
    stats = {
        "total": len(dataset),
        "processed": len(task_queue),
        "existing": len(existing_ids),
        "success": 0,
        "failed": 0,
        "test_model": args.test_model
    }
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_code_worker, task_args): task_args for task_args in task_queue}
        
        for future in as_completed(futures):
            task_bar.update(1)
            
            try:
                result = future.result()
                
                if result["success"]:
                    results.append(result)
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    # Also save the failed results
                    results.append(result)
                    
            except Exception as e:
                stats["failed"] += 1
                print(f"Task error: {e}")
            
            # Batch save
            if len(results) >= args.batch_size:
                write_jsonl_file(results, results_path, format="a")
                results.clear()
    
    # Save the remaining results
    if results:
        write_jsonl_file(results, results_path, format="a")
    
    task_bar.close()
    
    # Save the statistics
    stats_path = os.path.join(output_dir, "generation_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print the results
    print(f"\nCode generation complete for {args.test_model}!")
    print(f"Total dataset size: {stats['total']}")
    print(f"Already processed: {stats['existing']}")
    print(f"Newly processed: {stats['processed']}")
    print(f"Success: {stats['success']} ({stats['success']/stats['processed']*100:.1f}% of new)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['processed']*100:.1f}% of new)")
    print(f"\nResults saved to: {results_path}")
    print(f"Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()


