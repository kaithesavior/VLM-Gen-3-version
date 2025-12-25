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

def _step1_visual_analysis(frame_paths: List[str], fps: int) -> VisualAnalysisReport:
    """
    Step 1: Visual Understanding via VLM.
    Extracts scene semantics, objects, and activities.
    """
    print(f"[{MODEL_NAME}] Starting Step 1: Visual Analysis on {len(frame_paths)} frames...")
    
    parts = []
    
    # 1. System Prompt for Step 1
    prompt = f"""
    You are an expert Video Understanding Engine.
    I have provided a sequence of video frames extracted at {fps} FPS.
    
    YOUR TASK:
    Perform a strict VISUAL analysis of the video.
    
    REQUIREMENTS:
    1. **Scene Semantics**: Identify the setting (e.g., kitchen, forest, beach).
    2. **Object Detection**: List objects, especially those with potential olfactory properties (e.g., food, flowers, fire).
    3. **Action Recognition**: Describe actions and interactions (e.g., cutting, boiling, burning).
    4. **Visual State Tracking**: Note the physical state of objects (e.g., "whole onion" -> "chopped onion").
    
    CONSTRAINTS:
    - **DO NOT** infer smells or chemical molecules yet.
    - Focus ONLY on what is visually observable.
    - Return key frames where visual state changes occur.
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
            
        return VisualAnalysisReport.model_validate_json(response.text)
        
    except Exception as e:
        print(f"Step 1 Failed: {e}")
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
    
    REQUIREMENTS:
    1. **Analyze Visual Evidence**: For each frame/event, look for "olfactory triggers" (e.g., peeling an orange, rain on dry earth).
    2. **Progressive Scent Analysis**: 
       - Track the **evolution** of the scent. Does it start faint and become strong?
       - Capture **intermediate stages** (e.g., initial cut vs. deep exposure vs. full release).
       - Ensure the `intensity` and `descriptors` reflect these changes dynamically.
    3. **Chemical Mapping**: 
       - Identify specific candidate molecules responsible for the smell (e.g., Limonene, Geosmin, Guaiacol).
       - **Diversity**: As the scent intensifies, list more complex/secondary molecules if scientifically accurate (e.g., starting with just Limonene, then adding Citral, Neral, Geranial as more cells rupture).
    4. **Fill the 'scent' field**: The input JSON has no scent data. You must ADD it.
    
    GUIDELINES FOR "LEMON/CITRUS" EXAMPLES (Reference Standard):
    - **Initial Cut**: Low intensity. Primary molecule: Limonene. Descriptor: Faint, Zesty.
    - **Deep Cut/Juice Exposure**: Medium to High intensity. Molecules: Limonene + Citral + beta-Pinene. Descriptors: Acidic, Tart, Fresh.
    - **Full Separation/Squeezing**: High/Strong intensity. Molecules: Limonene + Citral + Neral + Geranial + Linalool. Descriptors: Sharp, Intense, Fruity, Floral notes.
    
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
