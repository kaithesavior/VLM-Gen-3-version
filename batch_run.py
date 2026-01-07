import os
import subprocess
import time

def run_batch():
    video_dir = "test_videos"
    files = sorted([f for f in os.listdir(video_dir) if f.startswith("test video") and f.endswith(".mp4")])
    
    # Sort by number correctly (1, 2, ..., 10, not 1, 10, 11...)
    def get_num(name):
        try:
            return int(name.replace("test video ", "").replace(".mp4", ""))
        except:
            return 999
            
    files.sort(key=get_num)
    
    # Filter for 11-15
    target_files = [f for f in files if 11 <= get_num(f) <= 15]
    
    print(f"Found {len(target_files)} videos to process: {target_files}")
    
    for filename in target_files:
        path = os.path.join(video_dir, filename)
        print(f"\n\n{'='*50}")
        print(f"PROCESSING: {filename}")
        print(f"{'='*50}\n")
        
        start_time = time.time()
        try:
            # Run main.py
            # blocking call
            result = subprocess.run(
                ["python3", "main.py", path],
                capture_output=False, # Let it stream to stdout
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ ERROR processing {filename}. Exit code: {result.returncode}")
            else:
                print(f"✅ SUCCESS processing {filename}")
                
        except Exception as e:
            print(f"❌ EXCEPTION processing {filename}: {e}")
            
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f}s")
        
        # Small cooldown to be nice to API
        time.sleep(2)

if __name__ == "__main__":
    run_batch()
