import json
import glob
import os

files = sorted(glob.glob("output_reports/test video 5_analysis_*.json"))[-3:]
print(f"Analyzing last 3 reports: {[os.path.basename(f) for f in files]}")

for f in files:
    print(f"\n--- Report: {os.path.basename(f)} ---")
    with open(f) as json_file:
        data = json.load(json_file)
        events = data.get("olfactory_events", [])
        
        dog_events = [e for e in events if "dog" in e["evidence_ref"]["object_name"].lower() or "retriever" in e["evidence_ref"]["object_name"].lower()]
        
        # 1. Check Hard Constraint: Max Duration <= 4.0s
        violation = False
        for e in dog_events:
            t = e["evidence_ref"]["interval_time"]
            duration = t["end_s"] - t["start_s"]
            activity = e.get("intensity", {}).get("categorical_level", "unknown")
            print(f"  Interval: {t['start_s']}-{t['end_s']} ({duration:.2f}s) | Level: {activity} | Intensity: {e['intensity']['numeric_level']}")
            
            if duration > 4.0 and activity in ["medium", "high"]:
                 print(f"    [VIOLATION] Duration {duration:.2f}s > 4.0s for {activity} activity!")
                 violation = True
        
        if not violation:
            print("  [PASS] All high-activity intervals <= 4.0s")
            
        # 2. Check Intensity Curve
        intensities = [e["intensity"]["numeric_level"] for e in dog_events]
        print(f"  Intensity Curve: {intensities}")
