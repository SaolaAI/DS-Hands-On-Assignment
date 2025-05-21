from pydantic import BaseModel, conlist, StrictStr
from typing import Literal, Dict, List

class ClusterRequest(BaseModel):
    data: conlist(Dict, min_length=100)
    optimization_metric: Literal["Silhouette", "Davies-Bouldin"]

class ClusterResponse(BaseModel):
    version: str
    metrics: Dict[str, float]
    visualization: str = None
