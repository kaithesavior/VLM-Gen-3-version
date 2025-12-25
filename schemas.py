from typing import List, Optional
from pydantic import BaseModel, Field

class VisualObject(BaseModel):
    name: str = Field(..., description="Object name")
    visual_state: str = Field(..., description="Physical state (e.g., sliced, boiling)")
    thermal_cue: str = Field(..., description="Heat evidence (e.g., steam)")
    interaction: str = Field(..., description="Action involving this object")
    location: str = Field(..., description="Position in scene")

class ScentInference(BaseModel):
    category: str = Field(..., description="Scent family (e.g., Citrus)")
    descriptors: List[str] = Field(..., description="Adjectives (e.g., zesty)")
    molecules: List[str] = Field(..., description="Chemical compounds (e.g., Limonene)")
    intensity: str = Field(..., description="Strength level")
    reasoning: str = Field(..., description="Why this scent is inferred")

class FrameAnalysis(BaseModel):
    timestamp: float
    frame_id: str
    scene: str
    objects: List[VisualObject]
    scent: Optional[ScentInference]

class TimelineEvent(BaseModel):
    time: str
    event: str
    change: str

class VideoAnalysisReport(BaseModel):
    meta: dict
    visual_timeline: List[TimelineEvent]
    frame_log: List[FrameAnalysis]

# --- VOS Step 1 Specific Schemas ---

class VisualFrameAnalysis(BaseModel):
    """Visual-only analysis for Step 1 (No Scent)"""
    timestamp: float
    frame_id: str
    scene: str
    objects: List[VisualObject]

class VisualAnalysisReport(BaseModel):
    """Intermediate output for Step 1"""
    meta: dict
    visual_timeline: List[TimelineEvent]
    frame_log: List[VisualFrameAnalysis]
