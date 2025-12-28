"""EdgeFlow Traceability and Provenance System

This module provides comprehensive tracking of all transformations, optimizations,
and decisions made during the EdgeFlow compilation pipeline. It enables full
audit trails, debugging, and reproducibility of model optimizations.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """Types of transformations tracked in the system."""
    PARSING = "parsing"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    FUSION = "fusion"
    SCHEDULING = "scheduling"
    CODE_GENERATION = "code_generation"
    DEPLOYMENT = "deployment"
    BENCHMARKING = "benchmarking"


class SeverityLevel(Enum):
    """Severity levels for transformation events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationSeverity(Enum):
    """Severity levels for validation events.
    
    This is used specifically for validation events to provide more granular
    control over validation reporting and error handling.
    """
    INFO = "info"          # Informational message
    WARNING = "warning"    # Potential issue that doesn't block execution
    ERROR = "error"        # Error that should be addressed
    CRITICAL = "critical"  # Severe error that prevents further processing


@dataclass
class TransformationEvent:
    """Records a single transformation event in the pipeline."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    transformation_type: TransformationType = TransformationType.OPTIMIZATION
    component: str = ""
    description: str = ""
    input_artifacts: List[str] = field(default_factory=list)
    output_artifacts: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    severity: SeverityLevel = SeverityLevel.INFO
    error_message: Optional[str] = None
    source_location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "transformation_type": self.transformation_type.value,
            "component": self.component,
            "description": self.description,
            "input_artifacts": self.input_artifacts,
            "output_artifacts": self.output_artifacts,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "duration_ms": self.duration_ms,
            "severity": self.severity.value,
            "error_message": self.error_message,
            "source_location": self.source_location,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            transformation_type=TransformationType(data["transformation_type"]),
            component=data["component"],
            description=data["description"],
            input_artifacts=data["input_artifacts"],
            output_artifacts=data["output_artifacts"],
            parameters=data["parameters"],
            metrics=data["metrics"],
            duration_ms=data["duration_ms"],
            severity=SeverityLevel(data["severity"]),
            error_message=data.get("error_message"),
            source_location=data.get("source_location"),
        )


@dataclass
class ArtifactInfo:
    """Information about an artifact in the pipeline."""
    artifact_id: str
    name: str
    type: str  # "model", "config", "report", "code", etc.
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
        }


class ProvenanceTracker:
    """Tracks provenance and lineage of all artifacts and transformations."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.events: List[TransformationEvent] = []
        self.artifacts: Dict[str, ArtifactInfo] = {}
        self.start_time = datetime.now()
        self.metadata: Dict[str, Any] = {
            "edgeflow_version": "0.1.0",
            "python_version": "",
            "platform": "",
        }
        
        logger.info(f"Started provenance tracking session: {self.session_id}")
    
    def register_artifact(
        self,
        name: str,
        artifact_type: str,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> str:
        """Register a new artifact in the system."""
        artifact_id = str(uuid.uuid4())
        
        # Calculate file size and checksum if path provided
        size_bytes = None
        checksum = None
        if path and Path(path).exists():
            size_bytes = Path(path).stat().st_size
            # Simple checksum for now - could use SHA256 for production
            checksum = str(hash(Path(path).read_bytes()))
        
        artifact = ArtifactInfo(
            artifact_id=artifact_id,
            name=name,
            type=artifact_type,
            path=path,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata or {},
            created_by=created_by,
        )
        
        self.artifacts[artifact_id] = artifact
        logger.debug(f"Registered artifact: {name} ({artifact_id})")
        return artifact_id
    
    def start_transformation(
        self,
        transformation_type: TransformationType,
        component: str,
        description: str,
        input_artifacts: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        source_location: Optional[str] = None,
    ) -> str:
        """Start tracking a transformation."""
        event = TransformationEvent(
            transformation_type=transformation_type,
            component=component,
            description=description,
            input_artifacts=input_artifacts or [],
            parameters=parameters or {},
            source_location=source_location,
        )
        
        self.events.append(event)
        logger.debug(f"Started transformation: {description} ({event.event_id})")
        return event.event_id
    
    def complete_transformation(
        self,
        event_id: str,
        output_artifacts: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        duration_ms: Optional[float] = None,
        severity: SeverityLevel = SeverityLevel.INFO,
        error_message: Optional[str] = None,
    ) -> None:
        """Complete a transformation event."""
        event = next((e for e in self.events if e.event_id == event_id), None)
        if not event:
            logger.warning(f"Transformation event not found: {event_id}")
            return
        
        event.output_artifacts = output_artifacts or []
        event.metrics = metrics or {}
        event.duration_ms = duration_ms or 0.0
        event.severity = severity
        event.error_message = error_message
        
        logger.debug(f"Completed transformation: {event.description}")
    
    def log_event(
        self,
        transformation_type: TransformationType,
        component: str,
        description: str,
        input_artifacts: Optional[List[str]] = None,
        output_artifacts: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        severity: SeverityLevel = SeverityLevel.INFO,
        error_message: Optional[str] = None,
        source_location: Optional[str] = None,
    ) -> str:
        """Log a complete transformation event."""
        event = TransformationEvent(
            transformation_type=transformation_type,
            component=component,
            description=description,
            input_artifacts=input_artifacts or [],
            output_artifacts=output_artifacts or [],
            parameters=parameters or {},
            metrics=metrics or {},
            severity=severity,
            error_message=error_message,
            source_location=source_location,
        )
        
        self.events.append(event)
        logger.debug(f"Logged event: {description}")
        return event.event_id
    
    def get_artifact_lineage(self, artifact_id: str) -> List[TransformationEvent]:
        """Get the lineage of transformations that created or modified an artifact."""
        lineage = []
        for event in self.events:
            if artifact_id in event.input_artifacts or artifact_id in event.output_artifacts:
                lineage.append(event)
        return lineage
    
    def get_transformation_chain(self) -> List[TransformationEvent]:
        """Get the complete chain of transformations in chronological order."""
        return sorted(self.events, key=lambda e: e.timestamp)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the compilation session."""
        total_duration = sum(e.duration_ms for e in self.events)
        error_count = sum(1 for e in self.events if e.severity == SeverityLevel.ERROR)
        warning_count = sum(1 for e in self.events if e.severity == SeverityLevel.WARNING)
        
        transformation_counts = {}
        for event in self.events:
            t_type = event.transformation_type.value
            transformation_counts[t_type] = transformation_counts.get(t_type, 0) + 1
        
        return {
            "session_id": self.session_id,
            "total_events": len(self.events),
            "total_artifacts": len(self.artifacts),
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "warning_count": warning_count,
            "transformation_counts": transformation_counts,
            "session_start": self.start_time.isoformat(),
            "session_duration_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
        }
    
    def export_provenance_report(self, output_path: str) -> None:
        """Export complete provenance report to JSON file."""
        report = {
            "session_metadata": {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "metadata": self.metadata,
            },
            "artifacts": {aid: artifact.to_dict() for aid, artifact in self.artifacts.items()},
            "events": [event.to_dict() for event in self.events],
            "summary": self.get_summary_statistics(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exported provenance report to: {output_path}")
    
    def import_provenance_report(self, input_path: str) -> None:
        """Import provenance report from JSON file."""
        with open(input_path, 'r') as f:
            report = json.load(f)
        
        # Import session metadata
        session_meta = report["session_metadata"]
        self.session_id = session_meta["session_id"]
        self.start_time = datetime.fromisoformat(session_meta["start_time"])
        self.metadata = session_meta["metadata"]
        
        # Import artifacts
        self.artifacts = {}
        for aid, artifact_data in report["artifacts"].items():
            artifact_data["created_at"] = datetime.fromisoformat(artifact_data["created_at"])
            self.artifacts[aid] = ArtifactInfo(**artifact_data)
        
        # Import events
        self.events = [TransformationEvent.from_dict(event_data) for event_data in report["events"]]
        
        logger.info(f"Imported provenance report from: {input_path}")


class TraceabilityContext:
    """Context manager for tracking transformations with automatic timing."""
    
    def __init__(
        self,
        tracker: ProvenanceTracker,
        transformation_type: TransformationType,
        component: str,
        description: str,
        input_artifacts: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        source_location: Optional[str] = None,
    ):
        self.tracker = tracker
        self.transformation_type = transformation_type
        self.component = component
        self.description = description
        self.input_artifacts = input_artifacts
        self.parameters = parameters
        self.source_location = source_location
        self.event_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.output_artifacts: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.error_message: Optional[str] = None
    
    def __enter__(self) -> "TraceabilityContext":
        self.start_time = time.perf_counter()
        self.event_id = self.tracker.start_transformation(
            self.transformation_type,
            self.component,
            self.description,
            self.input_artifacts,
            self.parameters,
            self.source_location,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and self.event_id:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            severity = SeverityLevel.ERROR if exc_type else SeverityLevel.INFO
            error_message = str(exc_val) if exc_val else None
            
            self.tracker.complete_transformation(
                self.event_id,
                self.output_artifacts,
                self.metrics,
                duration_ms,
                severity,
                error_message,
            )
    
    def add_output_artifact(self, artifact_id: str) -> None:
        """Add an output artifact to this transformation."""
        self.output_artifacts.append(artifact_id)
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to this transformation."""
        self.metrics[name] = value


# Global tracker instance
_global_tracker: Optional[ProvenanceTracker] = None


def get_global_tracker() -> ProvenanceTracker:
    """Get or create the global provenance tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProvenanceTracker()
    return _global_tracker


def set_global_tracker(tracker: ProvenanceTracker) -> None:
    """Set the global provenance tracker."""
    global _global_tracker
    _global_tracker = tracker


def trace_transformation(
    transformation_type: TransformationType,
    component: str,
    description: str,
    input_artifacts: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    source_location: Optional[str] = None,
) -> TraceabilityContext:
    """Create a traceability context for a transformation."""
    tracker = get_global_tracker()
    return TraceabilityContext(
        tracker,
        transformation_type,
        component,
        description,
        input_artifacts,
        parameters,
        source_location,
    )


def register_artifact(
    name: str,
    artifact_type: str,
    path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    created_by: Optional[str] = None,
) -> str:
    """Register an artifact with the global tracker."""
    tracker = get_global_tracker()
    return tracker.register_artifact(name, artifact_type, path, metadata, created_by)


def export_session_report(output_path: str) -> None:
    """Export the current session's provenance report."""
    tracker = get_global_tracker()
    tracker.export_provenance_report(output_path)


if __name__ == "__main__":
    # Example usage
    tracker = ProvenanceTracker()
    
    # Register input model
    model_id = tracker.register_artifact(
        "input_model.tflite",
        "model",
        "/path/to/model.tflite",
        {"framework": "tensorflow", "size_mb": 5.2},
        "user",
    )
    
    # Track optimization
    with trace_transformation(
        TransformationType.QUANTIZATION,
        "optimizer",
        "INT8 quantization",
        input_artifacts=[model_id],
        parameters={"quantization_type": "int8", "calibration_dataset": "imagenet_subset"},
    ) as ctx:
        # Simulate optimization work
        time.sleep(0.1)
        
        # Register output
        optimized_id = tracker.register_artifact(
            "optimized_model.tflite",
            "model",
            "/path/to/optimized.tflite",
            {"framework": "tensorflow", "size_mb": 1.3, "quantized": True},
            "optimizer",
        )
        ctx.add_output_artifact(optimized_id)
        ctx.add_metric("size_reduction_percent", 75.0)
        ctx.add_metric("latency_improvement_ms", 12.5)
    
    # Export report
    tracker.export_provenance_report("provenance_report.json")
    print("Provenance tracking example completed!")
