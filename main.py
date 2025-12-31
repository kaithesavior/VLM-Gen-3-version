import argparse
import json
import os
import shutil
from datetime import datetime
from video_processor import extract_frames_to_folder
from vlm_client import analyze_video_sequence

def main():
    parser = argparse.ArgumentParser(description="Olfactory Video Analysis System (VOS Pipeline)")
    parser.add_argument("video_path", help="Path to the input video file")
    
    # Allow user to pass fps as a positional argument (optional)
    # e.g. python main.py video.mp4 10
    parser.add_argument("fps_pos", nargs="?", type=int, help="Frames per second to extract (positional)")
    
    parser.add_argument("--output", help="Path to save the JSON output", default=None)
    parser.add_argument("--fps", type=int, default=4, help="Frames per second to extract (flag)")
    
    args = parser.parse_args()
    
    # Logic: If positional fps is provided, use it; otherwise use flag or default
    target_fps = args.fps_pos if args.fps_pos is not None else args.fps
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return

    # Prepare output paths
    base = os.path.splitext(os.path.basename(args.video_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Ensure output_reports directory exists
    output_dir = "output_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.output:
        # If user specifies output path, use it directly (allow them to override folder)
        output_path = args.output
    else:
        # Default behavior: save inside output_reports folder
        output_path = os.path.join(output_dir, f"{base}_analysis_{timestamp}.json")
        
    temp_folder = os.path.join("temp_frames", f"{base}_{timestamp}")

    try:
        # Step 1: Extract Frames (Ground Truth)
        print("\n--- Step 1: Frame Extraction ---")
        print(f"Target FPS: {target_fps}")
        frame_paths = extract_frames_to_folder(args.video_path, temp_folder, target_fps)
        
        if not frame_paths:
            print("No frames extracted.")
            return

        # Step 2: VLM Sequence Analysis
        print("\n--- Step 2: VLM Sequence Analysis ---")
        report = analyze_video_sequence(frame_paths, target_fps)
        
        # Step 3: Save Report
        print("\n--- Step 3: Saving Report ---")
        # Add local metadata
        report.meta["source_video"] = args.video_path
        report.meta["generated_at"] = datetime.now().isoformat()
        
        with open(output_path, "w") as f:
            f.write(report.model_dump_json(indent=2))
            
        print(f"Success! Analysis saved to: {output_path}")
        print(f"Ground truth frames are preserved in: {temp_folder}")
        print("(You can use these frames to verify the JSON output visually)")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        
    # Note: We do NOT delete temp_folder automatically, 
    # because the user guideline says "These original frames are the benchmark for verification".

if __name__ == "__main__":
    main()
