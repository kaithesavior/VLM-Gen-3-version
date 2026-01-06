import argparse
import json
import os
import shutil
from datetime import datetime
from video_processor import extract_frames_to_folder
from vlm_client import perform_visual_analysis, perform_olfactory_inference

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
    
    # We will generate 3 files, so we modify the output path strategy
    if args.output:
        # If user provided a specific file, we will use it as a prefix/base
        # e.g. "result.json" -> "result_ours.json", "result_naive.json", etc.
        user_base, user_ext = os.path.splitext(args.output)
        path_ours = f"{user_base}_OURS{user_ext}"
        path_over = f"{user_base}_BASELINE_OVERINCLUSIVE{user_ext}"
        path_naive = f"{user_base}_BASELINE_NAIVE{user_ext}"
    else:
        # Default behavior
        path_ours = os.path.join(output_dir, f"{base}_OURS_{timestamp}.json")
        path_over = os.path.join(output_dir, f"{base}_BASELINE_OVERINCLUSIVE_{timestamp}.json")
        path_naive = os.path.join(output_dir, f"{base}_BASELINE_NAIVE_{timestamp}.json")
        
    temp_folder = os.path.join("temp_frames", f"{base}_{timestamp}")

    try:
        # Step 1: Extract Frames (Ground Truth)
        print("\n--- Step 1: Frame Extraction ---")
        print(f"Target FPS: {target_fps}")
        frame_paths = extract_frames_to_folder(args.video_path, temp_folder, target_fps)
        
        if not frame_paths:
            print("No frames extracted.")
            return

        # Step 3: Run 3 Experimental Conditions
        print("\n--- Step 2 & 3: Running Independent Pipelines ---")
        
        experiments = [
            ("OURS (System-generated Plan)", "step1_visual.txt", "step2_olfactory.txt", path_ours),
            ("BASELINE 1 (Over-Inclusive)", "step1_visual_overinclusive.txt", "step2_olfactory_overinclusive.txt", path_over),
            ("BASELINE 2 (Naive/Object-Based)", "step1_visual_naive.txt", "step2_olfactory_naive.txt", path_naive)
        ]
        
        for name, visual_prompt, olfactory_prompt, out_path in experiments:
            print(f"\n>>> Running Condition: {name}...")
            print(f"    Visual Prompt: {visual_prompt}")
            print(f"    Olfactory Prompt: {olfactory_prompt}")
            
            try:
                # Independent Step 1
                visual_report = perform_visual_analysis(frame_paths, target_fps, prompt_file=visual_prompt)
                
                # Independent Step 2
                report = perform_olfactory_inference(visual_report, prompt_file=olfactory_prompt)
                
                # Add local metadata
                report.meta["source_video"] = args.video_path
                report.meta["generated_at"] = datetime.now().isoformat()
                report.meta["experiment_condition"] = name
                
                with open(out_path, "w") as f:
                    f.write(report.model_dump_json(indent=2))
                print(f"    Saved to: {out_path}")
                
            except Exception as e:
                print(f"    FAILED Condition {name}: {e}")

        print(f"\nAll experiments complete!")
        print(f"Ground truth frames are preserved in: {temp_folder}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        
    # Note: We do NOT delete temp_folder automatically, 
    # because the user guideline says "These original frames are the benchmark for verification".

if __name__ == "__main__":
    main()
