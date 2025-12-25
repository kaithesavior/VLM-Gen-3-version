import os
import time
import json
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import VideoAnalysisReport, VisualAnalysisReport

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ.setdefault("GOOGLE_API_KEY", api_key or "")
client = genai.Client()

# Using 2.5-flash as it handles long context (many images) efficiently
MODEL_NAME = "gemini-2.5-flash"

def _step1_visual_analysis(frame_paths: List[str], fps: int, attempt: int = 1) -> VisualAnalysisReport:
    """
    Step 1: Visual Understanding via VLM.
    Extracts scene semantics, objects, and activities.
    """
    total_frames = len(frame_paths)
    estimated_duration = total_frames / fps
    expected_entries = int(estimated_duration)  # Expecting roughly 1 entry per second
    
    print(f"[{MODEL_NAME}] Starting Step 1: Visual Analysis on {total_frames} frames (Attempt {attempt})...")
    print(f"Estimated Video Duration: {estimated_duration:.2f}s. Expecting ~{expected_entries} log entries.")
    
    parts = []
    
    # 1. System Prompt for Step 1
    prompt = f"""
    You are an expert Video Understanding Engine.
    I have provided a sequence of video frames extracted at {fps} FPS.
    The TOTAL DURATION is exactly {estimated_duration:.2f} seconds.
    
    YOUR TASK:
    Perform a strict VISUAL analysis of the ENTIRE video sequence from 0.0s to {estimated_duration:.2f}s.
    
    CRITICAL REQUIREMENTS:
    1. **Full Timeline Coverage**: You MUST provide analysis that covers the ENTIRE duration. 
       - Do NOT stop at 4s or 7s if the video is {estimated_duration:.2f}s long.
       - The last entry in `frame_log` MUST be near {estimated_duration:.2f}s.
    2. **Dense Sampling**: You MUST provide a `frame_log` entry for roughly **EVERY 1.0 SECOND**.
       - I expect at least {expected_entries} entries in `frame_log`.
       - Do NOT summarize. List every second explicitly.
    3. **Scene Semantics**: Identify the setting.
    4. **Object Detection**: List objects.
    5. **Action Recognition**: Describe actions.
    6. **Visual State Tracking**: Note physical state changes.
    
    CONSTRAINTS:
    - **DO NOT** infer smells or chemical molecules yet.
    - Focus ONLY on what is visually observable.
    """
    parts.append(types.Part(text=prompt))
    
    # 2. Add Images
    for p in frame_paths:
        with open(p, "rb") as f:
            img_data = f.read()
        parts.append(types.Part(inline_data=types.Blob(data=img_data, mime_type="image/jpeg")))
        
    print("Sending visual data to VLM...")
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
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
                return _step1_visual_analysis(frame_paths, fps, attempt + 1)
            else:
                print("CRITICAL: Max retries reached. Proceeding with incomplete data.")
                
        return report
        
    except Exception as e:
        print(f"Step 1 Failed: {e}")
        if attempt < 3:
             print(f"Retry triggered on error! Starting attempt {attempt + 1}...")
             return _step1_visual_analysis(frame_paths, fps, attempt + 1)
        raise e

def _step2_olfactory_inference(visual_report: VisualAnalysisReport) -> VideoAnalysisReport:
    """
    Step 2: Semantic-to-Chemical Translation via LLM.
    Maps visual semantics to olfactory representations.
    """
    print(f"[{MODEL_NAME}] Starting Step 2: Olfactory Inference (LLM)...")
    
    # Convert visual report to JSON string for the prompt
    visual_json = visual_report.model_dump_json(indent=2)
    
    prompt = f"""
    You are a Computational Olfactory Expert (Digital Scent Chemist).
    
    INPUT:
    A structured visual analysis of a video (JSON format), describing scenes, objects, and actions.
    
    YOUR TASK:
    Transform this visual description into a structured olfactory representation.
    
    RULES & GUIDELINES (Strict Enforcement):
    
    1. **Dynamic Intensity Mapping**:
       - **Low Intensity**: Initial contact, surface-level actions, or closed containers.
       - **Medium Intensity**: Breaking of skin/peel, heating begins, or partial exposure.
       - **High Intensity**: Full rupture, boiling, frying, or active squeezing/spreading.
       
    2. **Molecular Complexity Progression**:
       - **Early Stage**: List only primary volatiles (e.g., Limonene for citrus).
       - **Late Stage**: MUST include secondary and trace compounds (e.g., Citral, Neral, Geranial, Linalool).
       - **Reaction Products**: If heat is involved, MUST list reaction products (e.g., Maillard compounds like Pyrazines).
       
    3. **Descriptor Evolution**:
       - Adjectives MUST change over time.
       - Start with generic terms (Fresh, Faint).
       - Evolve into specific terms (Tart, Sharp, Caramelized, Smoky).
       
    4. **Causal Reasoning**:
       - The 'reasoning' field must explain the PHYSICAL cause of the scent change.
       - Example: "Cellular rupture of the flavedo releases essential oils" is better than "It smells like lemon".

    OUTPUT FORMAT:
    Return the COMPLETE JSON matching the 'VideoAnalysisReport' schema.
    This schema is identical to the input but includes the 'scent' field for each frame in 'frame_log'.
    
    INPUT DATA:
    {visual_json}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": VideoAnalysisReport.model_json_schema()
            }
        )
        
        if not response.text:
            raise ValueError("Empty response from LLM Step 2")
            
        return VideoAnalysisReport.model_validate_json(response.text)
        
    except Exception as e:
        print(f"Step 2 Failed: {e}")
        raise e

def analyze_video_sequence(frame_paths: List[str], fps: int) -> VideoAnalysisReport:
    """
    Orchestrates the 2-step VOS pipeline.
    """
    # Step 1: Visual Analysis
    visual_report = _step1_visual_analysis(frame_paths, fps)
    print("Step 1 Complete. Visual Timeline extracted.")
    
    # Step 2: Olfactory Inference
    final_report = _step2_olfactory_inference(visual_report)
    print("Step 2 Complete. Chemical mapping finished.")
    
    return final_report
