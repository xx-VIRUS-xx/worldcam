"""Face identity: FaceAnalyzer (InsightFace, loaded lazily) and IdentityResolver."""
from .faces import FaceAnalyzer
from .resolver import Candidate, Decision, IdentityResolver, ResolveResult

__all__ = ["FaceAnalyzer", "IdentityResolver", "ResolveResult", "Decision", "Candidate"]
