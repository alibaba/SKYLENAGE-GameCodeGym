from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import jsonlines
import argparse
import tqdm
import traceback
import tempfile
import time
import cv2
import numpy as np
from PIL import Image
import pygame
import threading
import queue
from typing import List, Dict, Any

# Utility functions
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

def create_video_from_frames_fast(frames: List[np.ndarray], output_path: str, fps: int = 25) -> bool:
    """Fast batch video creation with optimized settings"""
    if not frames:
        return False
    
    try:
        height, width = frames[0].shape[:2]
        
        # Use faster codec settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Batch write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def run_code_with_screenshots_and_video_optimized(code_text: str, output_dir: str, game_id: str, 
                                                 record_duration: int = 10, video_fps: int = 3, 
                                                 screenshot_format: str = "jpg", async_io: bool = True):
    """Optimized version with all performance improvements"""
    screenshot_dir = os.path.join(output_dir, "screenshots")
    video_dir = os.path.join(output_dir, "videos")
    ensure_directory_exists(screenshot_dir, type="dir")
    ensure_directory_exists(video_dir, type="dir")
    
    # Calculate frame interval for lower fps
    frame_interval = 1.0 / video_fps  # e.g., 3fps = 0.333s interval
    
    screenshot_code = f"""
import pygame
import sys
import os
import time
import cv2
import numpy as np
import atexit
import threading
import queue
from PIL import Image

# Create directories
screenshot_dir = "{screenshot_dir}"
video_dir = "{video_dir}"
os.makedirs(screenshot_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Configuration
ASYNC_IO = {async_io}
VIDEO_FPS = {video_fps}
FRAME_INTERVAL = {frame_interval}
SCREENSHOT_FORMAT = "{screenshot_format}"
RECORD_DURATION = {record_duration}

# Save the original display functions
_original_update = pygame.display.update
_original_flip = pygame.display.flip

_start_time = None
_screenshot_times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_screenshots_taken = set()
_video_frames = []
_last_frame_time = 0

# Async I/O components
_io_queue = queue.Queue()
_io_thread = None
_io_running = True

def _pygame_surface_to_cv2_optimized(surface):
    '''Optimized pygame surface to cv2 conversion with minimal copying'''
    try:
        # Get size
        w, h = surface.get_size()
        
        # Use tobytes instead of tostring for better performance
        raw = pygame.image.tobytes(surface, 'RGB')
        
        # Create numpy array without unnecessary copying
        img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        
        # Convert RGB to BGR in-place when possible
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    except Exception as e:
        print(f"Error in surface conversion: {{e}}")
        return None

def _save_screenshot_optimized(surface, filepath, format_type="jpg"):
    '''Optimized screenshot saving'''
    try:
        if format_type.lower() == "jpg":
            # Convert to PIL for JPEG compression
            w, h = surface.get_size()
            raw = pygame.image.tobytes(surface, 'RGB')
            img = Image.frombuffer('RGB', (w, h), raw)
            img.save(filepath, 'JPEG', quality=85, optimize=True)
        else:
            pygame.image.save(surface, filepath)
        return True
    except Exception as e:
        print(f"Error saving screenshot: {{e}}")
        return False

def _io_worker():
    '''Background thread for async I/O operations'''
    while _io_running:
        try:
            task = _io_queue.get(timeout=1)
            if task is None:  # Poison pill
                break
                
            task_type = task['type']
            if task_type == 'screenshot':
                _save_screenshot_optimized(task['surface'], task['filepath'], task['format'])
            elif task_type == 'frame':
                # Frame data is already processed, just store it
                _video_frames.append(task['frame'])
                
            _io_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"IO worker error: {{e}}")

def _start_io_thread():
    global _io_thread
    if ASYNC_IO and _io_thread is None:
        _io_thread = threading.Thread(target=_io_worker, daemon=True)
        _io_thread.start()

def _capture_frame_and_screenshot():
    global _start_time, _screenshots_taken, _last_frame_time
    
    if _start_time is None:
        _start_time = time.time()
        _start_io_thread()
    
    current_time = time.time()
    elapsed_time = current_time - _start_time
    
    # Stop recording after duration
    if elapsed_time >= RECORD_DURATION:
        return
    
    screen = pygame.display.get_surface()
    if not screen:
        return
    
    # Take screenshots at specific times
    for screenshot_time in _screenshot_times:
        if elapsed_time >= screenshot_time and screenshot_time not in _screenshots_taken:
            filepath = os.path.join(screenshot_dir, f"time_{{screenshot_time:02d}}s.{{SCREENSHOT_FORMAT}}")
            
            if ASYNC_IO:
                # Async screenshot saving
                _io_queue.put({{
                    'type': 'screenshot',
                    'surface': screen.copy(),  # Make a copy for async processing
                    'filepath': filepath,
                    'format': SCREENSHOT_FORMAT
                }})
            else:
                # Sync screenshot saving
                _save_screenshot_optimized(screen, filepath, SCREENSHOT_FORMAT)
            
            _screenshots_taken.add(screenshot_time)
    
    # Record video frames with optimized fps
    if elapsed_time - _last_frame_time >= FRAME_INTERVAL:
        try:
            cv2_frame = _pygame_surface_to_cv2_optimized(screen)
            if cv2_frame is not None:
                if ASYNC_IO:
                    # Store frame directly (already converted)
                    _io_queue.put({{
                        'type': 'frame',
                        'frame': cv2_frame.copy()
                    }})
                else:
                    _video_frames.append(cv2_frame)
                _last_frame_time = elapsed_time
        except Exception as e:
            print(f"Error capturing frame: {{e}}")

def _custom_update(*args, **kwargs):
    _capture_frame_and_screenshot()
    return _original_update(*args, **kwargs)

def _custom_flip():
    _capture_frame_and_screenshot()
    return _original_flip()

# Replace the display functions
pygame.display.update = _custom_update
pygame.display.flip = _custom_flip

def _cleanup():
    global _video_frames, _io_running, _io_thread
    
    # Stop async I/O
    if ASYNC_IO and _io_thread:
        _io_running = False
        _io_queue.put(None)  # Poison pill
        _io_thread.join(timeout=5)
    
    # Create video with batch processing
    if _video_frames:
        try:
            video_path = os.path.join(video_dir, "gameplay.mp4")
            if len(_video_frames) > 0:
                height, width = _video_frames[0].shape[:2]
                
                # Fast video creation with optimized settings
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                actual_fps = VIDEO_FPS
                out = cv2.VideoWriter(video_path, fourcc, actual_fps, (width, height))
                
                # Batch write all frames
                for frame in _video_frames:
                    out.write(frame)
                
                out.release()
                
                actual_duration = len(_video_frames) / actual_fps
                print(f"Video saved: {{len(_video_frames)}} frames at {{actual_fps}}fps = {{actual_duration:.1f}}s (recorded at {{VIDEO_FPS}}fps)")
        except Exception as e:
            print(f"Error saving video: {{e}}")

atexit.register(_cleanup)

# Original code starts here
"""
    
    # Combine the code
    full_code = screenshot_code + "\n" + code_text
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        # Set environment variables for better performance
        env = os.environ.copy()
        env['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        
        # Run the code with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=record_duration + 15,  # Increased buffer for cleanup
            env=env
        )
        
        # Check generated media files
        screenshots = []
        if os.path.exists(screenshot_dir):
            for f in sorted(os.listdir(screenshot_dir)):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    screenshots.append(os.path.join(screenshot_dir, f))
        
        video_path = os.path.join(video_dir, "gameplay.mp4")
        video_exists = os.path.exists(video_path) and os.path.getsize(video_path) > 0
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "screenshots": screenshots,
            "video_path": video_path if video_exists else None,
            "video_exists": video_exists,
            "optimization_stats": {
                "video_fps": video_fps,
                "screenshot_format": screenshot_format,
                "async_io_enabled": async_io,
                "frame_interval": frame_interval
            }
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timeout (exceeded {record_duration + 15} seconds)",
            "screenshots": [],
            "video_path": None,
            "video_exists": False,
            "optimization_stats": {}
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "screenshots": [],
            "video_path": None,
            "video_exists": False,
            "optimization_stats": {}
        }
    finally:
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file)
            except:
                pass

def task_worker(task_args: Dict[str, Any]) -> Dict[str, Any]:
    """Optimized task worker with performance improvements"""
    dataset_item = task_args.get("dataset_item", {})
    output_dir = task_args.get("output_dir", "./recording_results")
    record_duration = task_args.get("record_duration", 10)
    video_fps = task_args.get("video_fps", 3)
    screenshot_format = task_args.get("screenshot_format", "jpg")
    async_io = task_args.get("async_io", True)
    
    game_id = str(dataset_item.get("game_id", "unknown"))
    requirement = dataset_item.get("requirement", "")
    reference_code = dataset_item.get("reference_code", "")
    metadata = dataset_item.get("metadata", {})
    test_model = dataset_item.get("test_model", "unknown")
    generated_code = dataset_item.get("generated_code", "")
    
    original_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory(prefix=f"pygame_game_{game_id}_") as temp_work_dir:
        try:
            # Switch to temporary directory
            os.chdir(temp_work_dir)
            print(f"Working in temp dir: {temp_work_dir}")
            
            # Create game-specific output directory (under the original output path)
            game_output_dir = os.path.join(output_dir, game_id)
            ensure_directory_exists(game_output_dir, type="dir")

            # Run the optimized code recording
            run_result = run_code_with_screenshots_and_video_optimized(
                generated_code, 
                game_output_dir, 
                game_id, 
                record_duration,
                video_fps,
                screenshot_format,
                async_io
            )
            
            # Build the result
            result = {
                "success": True,
                "game_id": game_id,
                "requirement": requirement,
                "generated_code": generated_code,
                "reference_code": reference_code,
                "test_model": test_model,
                "execution_result": {
                    "success": run_result["success"],
                    "stdout": run_result["stdout"][:500],
                    "stderr": run_result["stderr"][:500],
                    "screenshot_count": len(run_result["screenshots"]),
                    "video_exists": run_result["video_exists"]
                },
                "media_files": {
                    "screenshots": run_result["screenshots"],
                    "video_path": run_result["video_path"],
                    "screenshot_dir": os.path.join(game_output_dir, "screenshots"),
                    "video_dir": os.path.join(game_output_dir, "videos")
                },
                "optimization_stats": run_result.get("optimization_stats", {}),
                "recording_duration": record_duration,
                "timestamp": time.time(),
                "temp_work_dir": temp_work_dir  # Record temporary directory (for debugging)
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "game_id": game_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "test_model": test_model,
                "optimization_stats": {}
            }
        finally:
            # Ensure switching back to the original working directory
            try:
                os.chdir(original_cwd)
            except Exception as e:
                print(f"Warning: Failed to restore working directory: {e}")
    # tempfile.TemporaryDirectory will automatically clean up the temporary directory

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized pygame recording with screenshots and videos")
    parser.add_argument("--generated_codes_path", type=str, required=True,
                       help="Path to the pre-generated codes")
    parser.add_argument("--output_path", type=str, 
                       default="./recording_results/")
    parser.add_argument("--test_model", type=str, required=True, 
                       help="Name of the model to test")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--record_duration", type=int, default=30,
                       help="Recording duration in seconds")
    parser.add_argument("--video_fps", type=int, default=3,
                       help="Video recording fps (lower = faster processing)")
    parser.add_argument("--screenshot_format", type=str, default="jpg", 
                       choices=["jpg", "png"],
                       help="Screenshot format (jpg is faster)")
    parser.add_argument("--async_io", action="store_true", default=True,
                       help="Enable async I/O for better performance")
    parser.add_argument("--no_async_io", action="store_true",
                       help="Disable async I/O")
    parser.add_argument("--resume", action="store_true", 
                       help="Continue from the last interrupted point")
    args = parser.parse_args()
    
    # Handle async_io flag
    if args.no_async_io:
        args.async_io = False
    
    return args

def main():
    args = parse_args()
    
    print(f"🚀 Optimized Recording for model: {args.test_model}")
    print(f"📹 Recording duration: {args.record_duration}s")
    print(f"🎬 Video FPS: {args.video_fps}")
    print(f"📸 Screenshot format: {args.screenshot_format}")
    print(f"⚡ Async I/O: {'Enabled' if args.async_io else 'Disabled'}")
    print(f"👥 Workers: {args.workers}")
    
    # Load the pre-generated codes
    generated_codes_path = os.path.join(
        args.generated_codes_path, 
        args.test_model.replace("/", "_"), 
        "generated_codes.jsonl"
    )
    print(f"Loading generated codes from {generated_codes_path}")
    dataset = read_jsonl_file(generated_codes_path, max_sentence=args.max_samples)
    
    # Filter the successfully generated codes
    dataset = [item for item in dataset if item.get("success", False)]
    print(f"Found {len(dataset)} successfully generated codes")
    
    if args.start_from > 0:
        dataset = dataset[args.start_from:]
    
    # Create the output directory
    output_dir = os.path.join(args.output_path, args.test_model.replace("/", "_"))
    ensure_directory_exists(output_dir, type="dir")
    
    # Output file path
    results_path = os.path.join(output_dir, "recorded_media.jsonl")
    
    # If resume mode, load the existing results
    existing_ids = set()
    if args.resume and os.path.exists(results_path):
        existing_results = read_jsonl_file(results_path)
        existing_ids = {r["game_id"] for r in existing_results if r.get("success", False)}
        print(f"Resume mode: found {len(existing_ids)} existing results")
    
    # Build the optimized task queue
    task_queue = []
    for item in dataset:
        if str(item.get("game_id")) not in existing_ids:
            task_queue.append({
                "dataset_item": item,
                "output_dir": output_dir,
                "record_duration": args.record_duration,
                "video_fps": args.video_fps,
                "screenshot_format": args.screenshot_format,
                "async_io": args.async_io
            })
    
    print(f"Tasks to process: {len(task_queue)}")
    
    if len(task_queue) == 0:
        print("All tasks already completed!")
        return
    
    # Process the tasks with optimizations
    task_bar = tqdm.tqdm(total=len(task_queue), desc=f"🎮 Recording with {args.workers} workers")
    results = []
    
    # Statistics
    stats = {
        "total": len(dataset),
        "processed": len(task_queue),
        "existing": len(existing_ids),
        "success": 0,
        "failed": 0,
        "test_model": args.test_model,
        "optimization_settings": {
            "record_duration": args.record_duration,
            "video_fps": args.video_fps,
            "screenshot_format": args.screenshot_format,
            "async_io": args.async_io,
            "workers": args.workers
        },
        "start_time": time.time()
    }
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(task_worker, task_args): task_args for task_args in task_queue}
        
        for future in as_completed(futures):
            task_bar.update(1)
            
            try:
                result = future.result()
                
                if result["success"]:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                
                results.append(result)
                    
            except Exception as e:
                stats["failed"] += 1
                print(f"Task error: {e}")
            
            # Batch save for better I/O performance
            if len(results) >= args.batch_size:
                write_jsonl_file(results, results_path, format="a")
                results.clear()
    
    # Save the remaining results
    if results:
        write_jsonl_file(results, results_path, format="a")
    
    task_bar.close()
    
    # Calculate performance metrics
    stats["end_time"] = time.time()
    stats["total_duration"] = stats["end_time"] - stats["start_time"]
    stats["avg_time_per_task"] = stats["total_duration"] / stats["processed"] if stats["processed"] > 0 else 0
    
    # Save the statistics
    stats_path = os.path.join(output_dir, "recording_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print the results with performance info
    print(f"\n🎉 Recording complete for {args.test_model}!")
    print(f"📊 Performance Summary:")
    print(f"  Total codes: {stats['total']}")
    print(f"  Already recorded: {stats['existing']}")
    print(f"  Newly recorded: {stats['processed']}")
    print(f"  Success: {stats['success']} ({stats['success']/stats['processed']*100:.1f}% of new)")
    print(f"  Failed: {stats['failed']} ({stats['failed']/stats['processed']*100:.1f}% of new)")
    print(f"  Total time: {stats['total_duration']:.1f}s")
    print(f"  Average time per task: {stats['avg_time_per_task']:.1f}s")
    print(f"\n📁 Output:")
    print(f"  Results: {results_path}")
    print(f"  Statistics: {stats_path}")

if __name__ == "__main__":
    main()

"""

python evaluation/screenshot_recorder.py \
    --generated_codes_path evaluation_results/generated_game_codes/ \
    --output_path evaluation_results/recording_results/ \
    --test_model "xxx" \
    --workers 20 \
    --record_duration 10 \
    --video_fps 30 \
    --screenshot_format jpg \
    --async_io \
    --batch_size 100 \
    --resume


"""