import os
import time
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import VideoAnalysisReport

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ.setdefault("GOOGLE_API_KEY", api_key or "")
client = genai.Client()

# Using 2.5-flash as it handles long context (many images) efficiently
MODEL_NAME = "gemini-2.5-flash"

def analyze_video_sequence(frame_paths: List[str], fps: int) -> VideoAnalysisReport:
    """
    Uploads all frames as a sequence and requests a full analysis.
    """
    print(f"Preparing to upload {len(frame_paths)} frames to VLM...")
    
    parts = []
    
    # 1. Add System Prompt
    prompt = f"""
    You are an expert Olfactory Video Analyst.
    I have provided a sequence of video frames extracted at {fps} FPS.
    The filenames (frame_00000.jpg...) represent the time sequence.
    
    YOUR TASK:
    Perform a complete "Visual-to-Olfactory" analysis of this entire sequence.
    
    REQUIREMENTS:
    1. **Visual Understanding**: Track objects, states, and actions over time.
    2. **Olfactory Inference**: For each moment, infer the smell based on visual evidence (e.g., cut lemon -> citric acid release).
    3. **Chemical Mapping**: Map these smells to specific molecules (e.g., Limonene, Guaiacol).
    
    OUTPUT FORMAT:
    Return a SINGLE JSON object matching the provided schema.
    For 'frame_log', you do NOT need to return every single frame if nothing changed. 
    Return key frames where visual state or scent changes occur, plus periodic updates (e.g., every 1 second).
    """
    parts.append(types.Part(text=prompt))
    
    # 2. Add Images
    # Note: For large sequences, we should use File API, but for simplicity/speed in this demo 
    # we use inline data. Ensure video is short (<1 min).
    for p in frame_paths:
        with open(p, "rb") as f:
            img_data = f.read()
        parts.append(types.Part(inline_data=types.Blob(data=img_data, mime_type="image/jpeg")))
        
    print("Sending request to VLM (this may take 30-60 seconds)...")
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=types.Content(parts=parts),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": VideoAnalysisReport.model_json_schema()
            }
        )
        
        if not response.text:
            raise ValueError("Empty response from VLM")
            
        return VideoAnalysisReport.model_validate_json(response.text)
        
    except Exception as e:
        print(f"VLM Analysis Failed: {e}")
        # Return empty/error report structure if needed, or raise
        raise e
