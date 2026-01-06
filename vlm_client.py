import os
import time
import json
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import OlfactoryAnalysisReport, VisualAnalysisReport

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ.setdefault("GOOGLE_API_KEY", api_key or "")
client = genai.Client()

# Using 2.5-flash as it handles long context (many images) efficiently
# Default fallback if not specified in config
DEFAULT_MODEL = "gemini-2.5-flash"

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: config.json not found, using defaults.")
        return {}

def _generate_environmental_prompt(config: dict) -> tuple[str, str]:
    """
    Generates dynamic prompt sections based on config settings.
    Returns: (step1_instructions, step2_rules)
    """
    # Step 1 Config (Visual Perception)
    s1_config = config.get("step1_visual_config", {})
    s1_parts = []
    
    if s1_config.get("detect_temperature_cues"):
        s1_parts.append("- **Temperature Cues**: Look for steam, condensation, fire, boiling, or frozen surfaces.")
    if s1_config.get("detect_airflow_indicators"):
        s1_parts.append("- **Airflow Indicators**: Look for smoke direction, moving hair/fabric, wind effects.")
    if s1_config.get("detect_humidity_visuals"):
        s1_parts.append("- **Humidity/Moisture**: Note if the environment appears dry, humid, rainy, or steamy.")
    if s1_config.get("detect_spatial_context"):
        s1_parts.append("- **Spatial Context**: Explicitly state if the scene is indoors (small/large room) or outdoors (open air).")
        
    if s1_parts:
        step1_extra = "\n8. **Environmental Context** (Required by Config):\n   " + "\n   ".join(s1_parts)
    else:
        step1_extra = ""

    # Step 2 Config (Olfactory Physics)
    s2_config = config.get("step2_olfactory_config", {})
    s2_parts = []
    
    if s2_config.get("apply_thermodynamics"):
        s2_parts.append("- **Thermodynamics**: Heat INCREASES volatility and intensity. Cold SUPPRESSES it. Reaction products (e.g., Maillard) only appear with heat.")
    if s2_config.get("apply_aerodynamics"):
        s2_parts.append("- **Aerodynamics**: Wind/Airflow disperses scent (lowering local intensity) but carries it further (plume effect).")
    if s2_config.get("apply_hygrometry"):
        s2_parts.append("- **Hygrometry**: High humidity/steam carries scent molecules better, often increasing perceived intensity.")
    if s2_config.get("apply_spatial_concentration"):
        s2_parts.append("- **Spatial Concentration**: Confined spaces (indoors, jars) INCREASE intensity due to accumulation. Open spaces (outdoors) DILUTE scent.")

    if s2_parts:
        step2_extra = "\n5. **Environmental Physics** (Strictly Enforce):\n   " + "\n   ".join(s2_parts)
    else:
        step2_extra = ""
        
    return step1_extra, step2_extra

def _step1_visual_analysis(frame_paths: List[str], fps: int, attempt: int = 1, prompt_file: str = "step1_visual.txt") -> VisualAnalysisReport:
    """
    Step 1: Visual Understanding via VLM.
    Extracts scene semantics, objects, and activities.
    """
    config = load_config()
    model_name = config.get("step1_visual_config", {}).get("model_name", DEFAULT_MODEL)
    
    step1_extra, _ = _generate_environmental_prompt(config)
    
    total_frames = len(frame_paths)
    estimated_duration = total_frames / fps
    expected_entries = int(estimated_duration)  # Expecting roughly 1 entry per second
    
    print(f"[{model_name}] Starting Step 1: Visual Analysis on {total_frames} frames (Attempt {attempt}) using {prompt_file}...")
    print(f"Estimated Video Duration: {estimated_duration:.2f}s. Expecting ~{expected_entries} log entries.")
    
    parts = []
    
    # 1. System Prompt for Step 1
    # Load prompt from external file
    try:
        with open(prompt_file, "r") as f:
            prompt_template = f.read()
            # Handle prompt formatting (some might not use all variables)
            try:
                prompt = prompt_template.format(
                    fps=fps,
                    estimated_duration=estimated_duration,
                    expected_entries=expected_entries,
                    extra_requirements=step1_extra
                )
            except KeyError:
                 # Fallback for baselines that might not need all vars
                 prompt = prompt_template.format(
                    fps=fps,
                    estimated_duration=estimated_duration
                )
    except Exception as e:
        print(f"Error loading Step 1 prompt from {prompt_file}: {e}")
        raise e
        
    parts.append(types.Part(text=prompt))
    
    # 2. Add Images
    for p in frame_paths:
        with open(p, "rb") as f:
            img_data = f.read()
        parts.append(types.Part(inline_data=types.Blob(data=img_data, mime_type="image/jpeg")))
        
    print("Sending visual data to VLM...")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=types.Content(parts=parts),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": VisualAnalysisReport.model_json_schema()
            }
        )
        
        if not response.text:
            raise ValueError("Empty response from VLM Step 1")
            
        report = VisualAnalysisReport.model_validate_json(response.text)
        
        # --- Validation Logic ---
        if not report.frame_log:
             print("WARNING: Step 1 returned empty frame_log.")
             valid_coverage = False
        else:
            last_timestamp = report.frame_log[-1].timestamp
            entry_count = len(report.frame_log)
            coverage_ratio = last_timestamp / estimated_duration
            
            print(f"Step 1 Output Coverage: {last_timestamp:.2f}s / {estimated_duration:.2f}s ({coverage_ratio:.1%})")
            print(f"Entry Count: {entry_count} / {expected_entries} expected")
            
            # Strict Validation Criteria
            # 1. Coverage must be at least 95% (Increased strictness)
            # 2. Entry count must be at least 80% of expected (allowing minor fps drift)
            if coverage_ratio >= 0.95 and entry_count >= (expected_entries * 0.8):
                valid_coverage = True
            else:
                valid_coverage = False
                print(f"WARNING: Output validation failed. Coverage: {coverage_ratio:.2f}, Entries: {entry_count}")

        if not valid_coverage:
            if attempt < 3:
                print(f"Retry triggered! Starting attempt {attempt + 1}...")
                return _step1_visual_analysis(frame_paths, fps, attempt + 1, prompt_file)
            else:
                print("CRITICAL: Max retries reached. Proceeding with incomplete data.")
                
        return report
        
    except Exception as e:
        print(f"Step 1 Failed: {e}")
        if attempt < 3:
             print(f"Retry triggered on error! Starting attempt {attempt + 1}...")
             return _step1_visual_analysis(frame_paths, fps, attempt + 1, prompt_file)
        raise e

def _step2_olfactory_inference(visual_report: VisualAnalysisReport, prompt_file: str = "step2_olfactory.txt") -> OlfactoryAnalysisReport:
    """
    Step 2: Semantic-to-Chemical Translation via LLM.
    Maps visual semantics to olfactory representations.
    """
    config = load_config()
    model_name = config.get("step2_olfactory_config", {}).get("model_name", DEFAULT_MODEL)
    
    _, step2_extra = _generate_environmental_prompt(config)
    
    print(f"[{model_name}] Starting Step 2: Olfactory Inference (LLM) using {prompt_file}...")
    
    # Convert visual report to JSON string for the prompt
    visual_json = visual_report.model_dump_json(indent=2)
    
    # Load prompt from external file
    try:
        with open(prompt_file, "r") as f:
            prompt_template = f.read()
            # Handle different prompt formatting needs
            # The standard prompt expects {visual_json} and {extra_rules}
            # Baseline prompts might only expect {visual_json}
            # We use safe substitution or try/except to handle this
            
            # Simple approach: Check format keys
            try:
                prompt = prompt_template.format(
                    visual_json=visual_json,
                    extra_rules=step2_extra
                )
            except KeyError:
                 # Fallback for baselines that might not have {extra_rules}
                 prompt = prompt_template.format(
                    visual_json=visual_json
                )

    except Exception as e:
        print(f"Error loading Step 2 prompt from {prompt_file}: {e}")
        raise e
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": OlfactoryAnalysisReport.model_json_schema()
            }
        )
        
        if not response.text:
            raise ValueError("Empty response from LLM Step 2")
            
        return OlfactoryAnalysisReport.model_validate_json(response.text)
        
    except Exception as e:
        print(f"Step 2 Failed: {e}")
        raise e

def perform_visual_analysis(frame_paths: List[str], fps: int, prompt_file: str = "step1_visual.txt") -> VisualAnalysisReport:
    """
    Public wrapper for Step 1.
    """
    return _step1_visual_analysis(frame_paths, fps, prompt_file=prompt_file)

def perform_olfactory_inference(visual_report: VisualAnalysisReport, prompt_file: str = "step2_olfactory.txt") -> OlfactoryAnalysisReport:
    """
    Public wrapper for Step 2.
    """
    return _step2_olfactory_inference(visual_report, prompt_file)

def analyze_video_sequence(frame_paths: List[str], fps: int) -> OlfactoryAnalysisReport:
    """
    Orchestrates the 2-step VOS pipeline (Standard "Ours" Mode).
    Kept for backward compatibility.
    """
    # Step 1: Visual Analysis
    visual_report = _step1_visual_analysis(frame_paths, fps)
    print("Step 1 Complete. Visual Timeline extracted.")
    
    # Step 2: Olfactory Inference (Default)
    final_report = _step2_olfactory_inference(visual_report)
    print("Step 2 Complete. Chemical mapping finished.")
    
    return final_report
