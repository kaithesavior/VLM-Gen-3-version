import cv2
import os
import shutil
from typing import List

def extract_frames_to_folder(video_path: str, output_folder: str, target_fps: int = 4) -> List[str]:
    """
    Extracts frames from video and saves them as sorted image files in a folder.
    
    Args:
        video_path (str): Path to input video.
        output_folder (str): Directory to save frames.
        target_fps (int): Frames per second to extract.
        
    Returns:
        List[str]: List of absolute paths to the saved image files, sorted by time.
    """
    # Clean/Create output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    print(f"Processing video: {video_path}")
    print(f"Original FPS: {original_fps:.2f}, Duration: {duration:.2f}s")
    
    frame_paths = []
    step_seconds = 1.0 / target_fps
    current_target_time = 0.0
    frame_count = 0
    
    while True:
        # Set position in milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, current_target_time * 1000)
        success, frame = cap.read()
        
        if not success:
            break
            
        # Save frame to disk
        # Naming convention: frame_00001.jpg ensures natural sorting
        filename = f"frame_{frame_count:05d}.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)
        
        frame_paths.append(os.path.abspath(filepath))
        
        current_target_time += step_seconds
        frame_count += 1
        
        # Stop if we exceed duration (add buffer)
        if current_target_time > duration + 0.1:
            break
            
    cap.release()
    print(f"Saved {len(frame_paths)} frames to {output_folder}")
    return frame_paths
