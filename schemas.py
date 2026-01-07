from typing import List, Optional
from pydantic import BaseModel, Field

class VisualObject(BaseModel):
    name: str = Field(..., description="Object name")
    proximity: str = Field(..., description="near, mid, far")
    frame_coverage: float = Field(..., description="Estimated screen area percentage (0.0-1.0)")
    motion: str = Field(..., description="Movement description relative to camera")
    visual_state_summary: str = Field(..., description="Brief state description")

class EnvironmentalFactors(BaseModel):
    temperature: Optional[str] = Field(None, description="Inferred ambient or object temp (e.g., 'Boiling hot', 'Iced')")
    airflow: Optional[str] = Field(None, description="Visual evidence of air movement (e.g., 'Steam rising vertically', 'Trees waving')")
    humidity: Optional[str] = Field(None, description="Inferred moisture level (e.g., 'Dry', 'Steamy', 'Rainy')")
    confinement: Optional[str] = Field(None, description="Spatial enclosure status (e.g., 'Open outdoors', 'Small closed room')")

class OdorSource(BaseModel):
    material_part: str = Field(..., description="e.g., peel, interior flesh, liquid surface")
    exposure_mode: str = Field(..., description="e.g., enclosed, exposed, wet, aerosol")

class IntensityModel(BaseModel):
    categorical_level: str = Field(..., description="low, medium, high")
    numeric_level: float = Field(..., description="0.0 to 1.0")
    trend: str = Field(..., description="ramp_up, plateau, ramp_down, pulsed")
    justification: Optional[str] = Field(None, description="Explanation for abrupt changes")

class MolecularProfile(BaseModel):
    primary_volatiles: List[str] = Field(..., description="3-6 main compounds/classes")
    secondary_trace: List[str] = Field(default=[], description="0-6 trace compounds")
    heat_reaction_products: List[str] = Field(default=[], description="Maillard products etc., empty if no heat")

class DescriptorEvolution(BaseModel):
    list: List[str] = Field(..., description="3-8 adjectives")
    descriptor_shift: Optional[str] = Field(None, description="Explanation of change vs previous state")
    shift_type: Optional[str] = Field(None, description="add, remove, intensify, soften, etc.")

class MixtureComponent(BaseModel):
    source_object_id: Optional[str] = None
    source_object_name: str
    weight: float

class UncertaintyModel(BaseModel):
    confidence: str = Field(..., description="high, medium, low")
    assumptions: List[str] = Field(default=[])

class NullScent(BaseModel):
    is_negligible: bool
    note: Optional[str] = None

class OlfactoryEvent(BaseModel):
    event_id: str = Field(..., description="Unique ID")
    evidence_ref: "EvidenceRef"
    odor_source: OdorSource
    intensity: IntensityModel
    molecular_profile: MolecularProfile
    descriptors: DescriptorEvolution
    reasoning: str
    mixture_attribution: List[MixtureComponent] = Field(default=[])
    uncertainty: UncertaintyModel
    null_scent: Optional[NullScent] = None

class ObjectTrack(BaseModel):
    object_name: str
    event_ids: List[str]

class FrameScentSampling(BaseModel):
    t_s: float
    linked_event_ids: List[str]
    interpolated_intensity: float
    dominant_descriptors: List[str]

class OlfactoryAnalysisReport(BaseModel):
    """Final Output Schema for Stage 2"""
    meta: dict
    olfactory_events: List[OlfactoryEvent]
    object_tracks: List[ObjectTrack]
    frame_scent_sampling: Optional[List[FrameScentSampling]] = None

class TimeInterval(BaseModel):
    start_s: float
    end_s: float

class ProximityData(BaseModel):
    category: str = Field(..., description="near, mid, far, exiting")
    frame_coverage_start: float
    frame_coverage_end: float
    trend: str = Field(..., description="stable, approaching, receding")

class EvidenceRef(BaseModel):
    object_name: str
    interval_time: TimeInterval
    proximity: Optional[ProximityData] = None

class VisualInterval(BaseModel):
    """
    Advanced interval-based representation for Step 1.
    Tracks a single smell-relevant object over a specific time range.
    """
    object_name: str = Field(..., description="Concise noun phrase")
    time: TimeInterval = Field(..., description="Start and end timestamps")
    visual_state: str = Field(..., description="Explicit physical description")
    state_change: bool = Field(..., description="True if this interval marks a new physical state")
    proximity: ProximityData = Field(..., description="Quantitative proximity metrics")
    activity_level: str = Field(..., description="low, medium, high")
    rationale: str = Field(..., description="Visual explanation of relevance to potential smell")

class FrameAnalysis(BaseModel):
    timestamp: float
    frame_id: str
    scene: str
    environment: Optional[EnvironmentalFactors] = Field(None, description="Environmental conditions affecting scent")
    objects: List[VisualObject]

# --- VOS Step 1 Specific Schemas ---

class VisualFrameAnalysis(BaseModel):
    """Visual-only analysis for Step 1 (No Scent)"""
    timestamp: float
    frame_id: str
    scene: str
    environment: Optional[EnvironmentalFactors] = Field(None, description="Environmental conditions affecting scent")
    objects: List[VisualObject]

class VisualAnalysisReport(BaseModel):
    """Intermediate output for Step 1"""
    meta: dict
    visual_timeline: List[VisualInterval]
    frame_log: List[VisualFrameAnalysis]
