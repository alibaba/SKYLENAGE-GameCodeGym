#!/usr/bin/env python3

import argparse
import base64
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import traceback

import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from tqdm import tqdm
import re

import re
import json

def extract_json_from_text(text):
    """
    Extract JSON content from text, handling various markdown and format issues
    """
    if not text:
        return ""
    
    # Convert to string (just in case)
    text = str(text).strip()
    
    # Method 1: Extract content from code blocks
    # Match ```json...``` or ```...```
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Method 2: If the entire text starts with ```
    if text.startswith('```'):
        # Remove ```json or ``` at the beginning
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
        # Remove ``` at the end
        text = re.sub(r'\n?```\s*$', '', text)
        return text.strip()
    
    # Method 3: Find JSON objects or arrays
    # Try to find the first { or [ to the last } or ]
    json_obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_obj_match:
        return json_obj_match.group(0)
    
    json_arr_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_arr_match:
        return json_arr_match.group(0)
    
    # If nothing is found, return the original text
    return text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    model_name: str
    vl_base_url: str
    text_base_url: str
    num_processes: int
    base_path: str = "evaluation_results"
    max_screenshots: int = 20  # Maximum number of screenshots to use
    max_retry_attempts: int = 5  # Maximum number of retry attempts for JSON parsing
    retry_delay: float = 1.0  # Delay between retries (seconds)
    use_streaming: bool = True  # Whether to use streaming request
    
    @property
    def code_path(self) -> Path:
        return Path(self.base_path) / "generated_game_codes" / self.model_name / "generated_codes.jsonl"
    
    @property
    def recording_path(self) -> Path:
        return Path(self.base_path) / "recording_results" / self.model_name
    
    @property
    def output_path(self) -> Path:
        return Path(self.base_path) / "evaluation_scores" / self.model_name
    
    @property
    def progress_file(self) -> Path:
        return self.output_path / "progress.json"


class GameEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.vl_client = OpenAI(api_key="EMPTY", base_url=config.vl_base_url)
        self.text_client = OpenAI(api_key="EMPTY", base_url=config.text_base_url)
        
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict:
        if self.config.progress_file.exists():
            with open(self.config.progress_file, 'r') as f:
                return json.load(f)
        return {"completed": [], "failed": {}}
    
    def save_progress(self):
        with open(self.config.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def load_game_codes(self) -> List[Dict]:
        games = []
        with open(self.config.code_path, 'r') as f:
            for idx, line in enumerate(f):
                game_data = json.loads(line)
                if 'game_id' not in game_data:
                    game_data['game_id'] = idx
                games.append(game_data)
        return games
    
    def encode_image_to_base64(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    
    def collect_stream_response(self, stream) -> str:
        collected_content = []
        try:
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_piece = chunk.choices[0].delta.content
                    collected_content.append(content_piece)
                    # Optional: print streaming content in real-time for debugging
                    # print(content_piece, end='', flush=True)
        except Exception as e:
            logger.error(f"Streaming response collection error: {e}")
            raise
        
        return ''.join(collected_content)
    
    def evaluate_code(self, game_data: Dict) -> Dict:
        requirement = game_data['requirement']
        code = game_data['generated_code']
        code_snippet = code 
        
        prompt = f"""As a code quality evaluation expert, please evaluate whether the following pygame game code meets the requirements.

Requirements: {requirement}

Generated code:
```python
{code_snippet}
```

Please evaluate the code from the following aspects (each 0-25 points):
1. Functionality (0-25 points): Whether all functions described in the requirements are implemented
2. Code Quality (0-25 points): Code structure, readability, and consistency
3. Game Logic (0-25 points): Whether the game logic is reasonable and complete
4. Technical Implementation (0-25 points): Whether pygame is used correctly and the algorithm efficiency

Please output JSON format:
{{
    "functionality_score": 0-25,
    "code_quality_score": 0-25,
    "game_logic_score": 0-25,
    "technical_score": 0-25,
    "total_score": 0-100,
    "feedback": "Detailed feedback",
    "checklist": {{
        "basic_pygame_setup": true/false,
        "game_loop_implemented": true/false,
        "event_handling": true/false,
        "graphics_rendering": true/false,
        "requirement_specific_features": true/false
    }}
}}

Output JSON directly, no other explanation:"""
        
        attempts = 0
        last_error = None
        
        while attempts < self.config.max_retry_attempts:
            attempts += 1
            result_text = ""
            
            try:
                logger.info(f"Code evaluation - Attempt {attempts}/{self.config.max_retry_attempts}")
                
                if self.config.use_streaming:
                    stream = self.text_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a professional code evaluation expert. Please ensure the output format is correct JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=4096,
                        stream=True  # Enable streaming response
                    )
                    
                    result_text = self.collect_stream_response(stream)
                else:
                    response = self.text_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a professional code evaluation expert. Please ensure the output format is correct JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=4096
                    )
                    result_text = response.choices[0].message.content
                
                extracted_json = extract_json_from_text(result_text)
                
                
                result = json.loads(extracted_json)
                
                required_fields = ["functionality_score", "code_quality_score", 
                                 "game_logic_score", "technical_score", "total_score"]
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                
                logger.info(f"Code evaluation successful - Attempt {attempts}")
                return result
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing error: {e}"
                logger.warning(f"Code evaluation failed - Attempt {attempts}: {last_error}")
                logger.debug(f"Original content: {result_text[:500]}...")  # Only record the first 500 characters
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Code evaluation failed - Attempt {attempts}: {last_error}")
            
            if attempts < self.config.max_retry_attempts:
                time.sleep(self.config.retry_delay)
        
        logger.error(f"Code evaluation failed, attempted {attempts} times: {last_error}")
        return {
            "error": f"Evaluation failed (attempt {attempts} times): {last_error}", 
            "total_score": 0,
            "attempts": attempts
        }
    
    def evaluate_screenshots(self, game_id: int, requirement: str) -> Dict:
        screenshot_dir = self.config.recording_path / str(game_id) / "screenshots"
        
        if not screenshot_dir.exists():
            return {"error": "Screenshot directory does not exist", "total_score": 0}
        
        screenshots = sorted(screenshot_dir.glob("*.jpg"))[:self.config.max_screenshots]
        
        if not screenshots:
            return {"error": "No screenshots found", "total_score": 0}
        
        content = [
            {
                "type": "text", 
                "text": f"""Based on {len(screenshots)} game screenshots, evaluate whether the game meets the following requirements:

Requirements: {requirement}

Please evaluate the game from the following aspects (each 0-25 points):
1. Visual Completeness (0-25 points): Whether the game screen is clear and the elements are complete
2. UI Design (0-25 points): UI layout, color matching, visual effects
3. Function Display (0-25 points): Whether the screenshots show the key functions in the requirements
4. Overall Quality (0-25 points): The overall visual quality and completion of the game

Please output JSON format:
{{
    "visual_completeness_score": 0-25,
    "ui_design_score": 0-25,
    "functionality_display_score": 0-25,
    "overall_quality_score": 0-25,
    "total_score": 0-100,
    "feedback": "Detailed feedback",
    "checklist": {{
        "game_window_visible": true/false,
        "graphics_rendered": true/false,
        "ui_elements_present": true/false,
        "game_content_visible": true/false,
        "visual_quality_acceptable": true/false
    }},
    "screenshot_analysis": "Analysis of the content of the screenshots"
}}

Output JSON directly, no other explanation:"""
            }
        ]
        
        # Add screenshots
        for screenshot_path in screenshots:
            content.append({
                "type": "image_url",
                "image_url": {"url": self.encode_image_to_base64(screenshot_path)}
            })
        
        attempts = 0
        last_error = None
        
        while attempts < self.config.max_retry_attempts:
            attempts += 1
            result_text = ""
            
            try:
                logger.info(f"Screenshot evaluation - Game {game_id} - Attempt {attempts}/{self.config.max_retry_attempts}")
                
                if self.config.use_streaming:
                    stream = self.vl_client.chat.completions.create(
                        model="/models/Qwen2.5-VL-72B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a professional game visual effect evaluation expert. Please ensure the output format is correct JSON."},
                            {"role": "user", "content": content}
                        ],
                        temperature=0.3,
                        max_tokens=4096,
                        stream=True  # Enable streaming response
                    )
                    
                    result_text = self.collect_stream_response(stream)
                else:
                    response = self.vl_client.chat.completions.create(
                        model="/models/Qwen2.5-VL-72B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a professional game visual effect evaluation expert. Please ensure the output format is correct JSON."},
                            {"role": "user", "content": content}
                        ],
                        temperature=0.3,
                        max_tokens=4096
                    )
                    result_text = response.choices[0].message.content
                
                extracted_json = extract_json_from_text(result_text)
                
                # Try to parse JSON
                result = json.loads(extracted_json)
                
                # Verify required fields exist
                required_fields = ["visual_completeness_score", "ui_design_score", 
                                 "functionality_display_score", "overall_quality_score", "total_score"]
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                
                logger.info(f"Screenshot evaluation successful - Game {game_id} - Attempt {attempts}")
                return result
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing error: {e}"
                logger.warning(f"Screenshot evaluation failed - Game {game_id} - Attempt {attempts}: {last_error}")
                logger.debug(f"Original content: {result_text[:500]}...")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Screenshot evaluation failed - Game {game_id} - Attempt {attempts}: {last_error}")
            
            if attempts < self.config.max_retry_attempts:
                time.sleep(self.config.retry_delay)
        
        logger.error(f"Screenshot evaluation failed - Game {game_id}, attempted {attempts} times: {last_error}")
        return {
            "error": f"Evaluation failed (attempt {attempts} times): {last_error}", 
            "total_score": 0,
            "attempts": attempts
        }
    
    def evaluate_video(self, game_id: int, requirement: str) -> Dict:
        video_path = self.config.recording_path / str(game_id) / "videos" / "gameplay.mp4"
        
        if not video_path.exists():
            return {"error": "Video file does not exist", "total_score": 0}
        
        video_size_mb = video_path.stat().st_size / (1024 * 1024)
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional game dynamic effect evaluation expert, good at analyzing the interactive logic and animation effects of games based on videos. Please ensure the output format is correct JSON."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Based on the provided game video, evaluate whether the game meets the following requirements:

Requirements: {requirement}

Video information: size {video_size_mb:.1f}MB

Please evaluate the game from the following aspects (each 0-25 points):
1. Animation Effect (0-25 points): Whether the animation of the game elements is smooth and natural
2. Interaction Logic (0-25 points): Whether the user interaction is correctly responded
3. Game Flow (0-25 points): Whether the game flow is complete and reasonable
4. Dynamic Quality (0-25 points): The overall dynamic effect and game playability

Please output JSON format:
{{
    "animation_score": 0-25,
    "interaction_score": 0-25,
    "gameplay_flow_score": 0-25,
    "dynamic_quality_score": 0-25,
    "total_score": 0-100,
    "feedback": "Detailed feedback",
    "checklist": {{
        "smooth_animation": true/false,
        "responsive_controls": true/false,
        "game_logic_working": true/false,
        "continuous_gameplay": true/false,
        "performance_acceptable": true/false
    }},
    "video_analysis": "Analysis of the content of the video"
}}

Output JSON directly, no other explanation:"""
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": f"file://{video_path}"}
                    }
                ]
            }
        ]
        
        attempts = 0
        last_error = None
        
        while attempts < self.config.max_retry_attempts:
            attempts += 1
            result_text = ""
            
            try:
                logger.info(f"Video evaluation - Game {game_id} - Attempt {attempts}/{self.config.max_retry_attempts}")
                
                if self.config.use_streaming:
                    stream = self.vl_client.chat.completions.create(
                        model="/models/Qwen2.5-VL-72B-Instruct",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4096,
                        stream=True,  # Enable streaming response
                        extra_body={
                            "mm_processor_kwargs": {
                                "fps": [2.0],
                                "min_pixels": 256 * 28 * 28,
                                "max_pixels": 1280 * 28 * 28,
                                "total_pixels": 20480 * 28 * 28
                            }
                        }
                    )
                    
                    result_text = self.collect_stream_response(stream)
                else:
                    response = self.vl_client.chat.completions.create(
                        model="/models/Qwen2.5-VL-72B-Instruct",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4096,
                        extra_body={
                            "mm_processor_kwargs": {
                                "fps": [2.0],
                                "min_pixels": 256 * 28 * 28,
                                "max_pixels": 1280 * 28 * 28,
                                "total_pixels": 20480 * 28 * 28
                            }
                        }
                    )
                    result_text = response.choices[0].message.content
                
                extracted_json = extract_json_from_text(result_text)
                
                result = json.loads(extracted_json)
                
                required_fields = ["animation_score", "interaction_score", 
                                 "gameplay_flow_score", "dynamic_quality_score", "total_score"]
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                
                logger.info(f"Video evaluation successful - Game {game_id} - Attempt {attempts}")
                return result
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing error: {e}"
                logger.warning(f"Video evaluation failed - Game {game_id} - Attempt {attempts}: {last_error}")
                logger.debug(f"Original content: {result_text[:500]}...")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Video evaluation failed - Game {game_id} - Attempt {attempts}: {last_error}")
            
            if attempts < self.config.max_retry_attempts:
                time.sleep(self.config.retry_delay)
        
        logger.error(f"Video evaluation failed - Game {game_id}, attempted {attempts} times: {last_error}")
        return {
            "error": f"Evaluation failed (attempt {attempts} times): {last_error}", 
            "total_score": 0,
            "attempts": attempts
        }
    
    def evaluate_single_game(self, game_id: int, game_data: Dict) -> Optional[Dict]:
        if game_id in self.progress["completed"]:
            logger.info(f"Game {game_id} has already been evaluated, skipping")
            return None
        
        logger.info(f"Starting evaluation for game {game_id}")
        
        try:
            result = {
                "game_id": game_id,
                "requirement": game_data["requirement"],
                "timestamp": datetime.now().isoformat(),
                "use_streaming": self.config.use_streaming  # Record whether to use streaming request
            }
            
            # 1. Evaluate code
            logger.info(f"Evaluating code - Game {game_id}")
            code_eval = self.evaluate_code(game_data)
            result["code_evaluation"] = code_eval
            
            # 2. Evaluate screenshots
            logger.info(f"Evaluating screenshots - Game {game_id}")
            screenshot_eval = self.evaluate_screenshots(game_id, game_data["requirement"])
            result["screenshot_evaluation"] = screenshot_eval
            
            # 3. Evaluate video
            logger.info(f"Evaluating video - Game {game_id}")
            video_eval = self.evaluate_video(game_id, game_data["requirement"])
            result["video_evaluation"] = video_eval
            
            total_score = 0
            score_count = 0
            for eval_type in ["code_evaluation", "screenshot_evaluation", "video_evaluation"]:
                if eval_type in result and "total_score" in result[eval_type]:
                    total_score += result[eval_type]["total_score"]
                    score_count += 1
            
            result["final_score"] = total_score / score_count if score_count > 0 else 0
            
            result["retry_info"] = {
                "code_attempts": result["code_evaluation"].get("attempts", 1),
                "screenshot_attempts": result["screenshot_evaluation"].get("attempts", 1),
                "video_attempts": result["video_evaluation"].get("attempts", 1)
            }
            
            output_file = self.config.output_path / f"game_{game_id}_evaluation.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.progress["completed"].append(game_id)
            self.save_progress()
            
            logger.info(f"Game {game_id} evaluation completed, total score: {result['final_score']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Game {game_id} evaluation failed: {e}")
            logger.error(traceback.format_exc())
            self.progress["failed"][str(game_id)] = str(e)
            self.save_progress()
            return None


def evaluate_game_worker(args):
    game_id, game_data, config = args
    evaluator = GameEvaluator(config)
    return evaluator.evaluate_single_game(game_id, game_data)


def main():
    parser = argparse.ArgumentParser(description="Game evaluation script")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--vl-base-url", required=True, help="Visual language model API address")
    parser.add_argument("--text-base-url", required=True, help="Pure text model API address")
    parser.add_argument("--num-processes", type=int, default=4, help="Number of concurrent processes")
    parser.add_argument("--start-id", type=int, default=0, help="Start game ID")
    parser.add_argument("--end-id", type=int, default=None, help="End game ID")
    parser.add_argument("--max-retry", type=int, default=10, help="Maximum number of retries when JSON parsing fails")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Delay between retries (seconds)")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming request")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        model_name=args.model_name,
        vl_base_url=args.vl_base_url,
        text_base_url=args.text_base_url,
        num_processes=args.num_processes,
        max_retry_attempts=args.max_retry,
        retry_delay=args.retry_delay,
        use_streaming=not args.no_streaming  # Determine whether to use streaming request based on command line arguments
    )
    
    # Create evaluator (for loading data)
    evaluator = GameEvaluator(config)
    
    # Load game data
    logger.info(f"Loading game code data: {config.code_path}")
    games = evaluator.load_game_codes()
    
    # Determine evaluation range
    end_id = args.end_id if args.end_id is not None else len(games)
    games_to_evaluate = [(i, game) for i, game in enumerate(games[args.start_id:end_id], start=args.start_id)]
    
    # Filter completed games
    games_to_evaluate = [(i, game) for i, game in games_to_evaluate if i not in evaluator.progress["completed"]]
    
    logger.info(f"Number of games to evaluate: {len(games_to_evaluate)}")
    logger.info(f"Maximum JSON parsing retry attempts: {config.max_retry_attempts}")
    logger.info(f"Retry delay: {config.retry_delay} seconds")
    logger.info(f"Use streaming request: {config.use_streaming}")
    
    if not games_to_evaluate:
        logger.info("All games have been evaluated")
        return
    
    
    tasks = [(game_id, game_data, config) for game_id, game_data in games_to_evaluate]
    
    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = {executor.submit(evaluate_game_worker, task): task[0] for task in tasks}
        
        with tqdm(total=len(tasks), desc="Evaluation progress") as pbar:
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    result = future.result(timeout=300)  # 5 minutes timeout
                    if result:
                        pbar.set_postfix({"Latest completed": f"Game {game_id}"})
                except Exception as e:
                    logger.error(f"Game {game_id} evaluation failed: {e}")
                finally:
                    pbar.update(1)
    
    # Generate summary report
    logger.info("Generating summary report...")
    generate_summary_report(config)
    logger.info("Evaluation completed!")


def generate_summary_report(config: EvaluationConfig):
    results = []
    
    # Read all evaluation results
    for result_file in config.output_path.glob("game_*_evaluation.json"):
        with open(result_file, 'r') as f:
            results.append(json.load(f))
    
    if not results:
        logger.warning("No evaluation results found")
        return
    
    # Calculate retry statistics
    retry_stats = {
        "code_avg_attempts": np.mean([r.get("retry_info", {}).get("code_attempts", 1) for r in results]),
        "screenshot_avg_attempts": np.mean([r.get("retry_info", {}).get("screenshot_attempts", 1) for r in results]),
        "video_avg_attempts": np.mean([r.get("retry_info", {}).get("video_attempts", 1) for r in results]),
    }
    
    # Calculate statistics
    summary = {
        "model_name": config.model_name,
        "total_games": len(results),
        "average_scores": {
            "final": np.mean([r["final_score"] for r in results]),
            "code": np.mean([r["code_evaluation"]["total_score"] for r in results if "total_score" in r.get("code_evaluation", {})]),
            "screenshot": np.mean([r["screenshot_evaluation"]["total_score"] for r in results if "total_score" in r.get("screenshot_evaluation", {})]),
            "video": np.mean([r["video_evaluation"]["total_score"] for r in results if "total_score" in r.get("video_evaluation", {})])
        },
        "score_distribution": {
            "excellent (80-100)": len([r for r in results if r["final_score"] >= 80]),
            "good (60-80)": len([r for r in results if 60 <= r["final_score"] < 80]),
            "fair (40-60)": len([r for r in results if 40 <= r["final_score"] < 60]),
            "poor (0-40)": len([r for r in results if r["final_score"] < 40])
        },
        "retry_statistics": retry_stats,
        "evaluation_config": {
            "max_retry_attempts": config.max_retry_attempts,
            "retry_delay": config.retry_delay,
            "use_streaming": config.use_streaming
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Save summary report
    summary_file = config.output_path / "summary_report.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary information
    print("\n" + "="*50)
    print(f"Evaluation summary - {config.model_name}")
    print("="*50)
    print(f"Total games: {summary['total_games']}")
    print(f"Average scores:")
    print(f"  - Total: {summary['average_scores']['final']:.2f}")
    print(f"  - Code: {summary['average_scores']['code']:.2f}")
    print(f"  - Screenshot: {summary['average_scores']['screenshot']:.2f}")
    print(f"  - Video: {summary['average_scores']['video']:.2f}")
    print(f"Score distribution:")
    for level, count in summary['score_distribution'].items():
        print(f"  - {level}: {count} ({count/summary['total_games']*100:.1f}%)")
    print(f"Retry statistics:")
    print(f"  - Code evaluation average retry: {retry_stats['code_avg_attempts']:.2f} times")
    print(f"  - Screenshot evaluation average retry: {retry_stats['screenshot_avg_attempts']:.2f} times")
    print(f"  - Video evaluation average retry: {retry_stats['video_avg_attempts']:.2f} times")
    print(f"Use streaming request: {config.use_streaming}")
    print("="*50)


if __name__ == "__main__":
    main()
