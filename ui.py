import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from sklearn.cluster import MiniBatchKMeans  # ← Changed import
from sklearn.metrics import silhouette_score
import numpy as np

app = FastAPI()

class FitRequest(BaseModel):
    data: List[Dict[str, float]]
    n_clusters: int
    batch_size: int = 1000  # ← Added MiniBatch parameter

@app.post("/fit")
async def fit_clusters(req: FitRequest):
    X = np.array([list(item.values()) for item in req.data])
    
    # Use MiniBatchKMeans with training parameters
    kmeans = MiniBatchKMeans(
        n_clusters=req.n_clusters,
        batch_size=req.batch_size,  # ← From request
        random_state=42  # ← Match training seed
    )
    labels = kmeans.fit_predict(X)

    silhouette = silhouette_score(X, labels)
    
    cluster_data = [dict(zip(req.data[0].keys(), row)) for row in X]
    
    similarity_edges = []
    for i in range(len(labels)-1):
        similarity_edges.append([i, i+1, np.linalg.norm(X[i]-X[i+1])])

    return {
        "metrics": {"silhouette": float(silhouette)},
        "cluster_labels": labels.tolist(),
        "cluster_data": cluster_data,
        "similarity_edges": similarity_edges
    }

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI"}

if __name__ == "__main__":
    host = os.environ.get("UVICORN_HOST", "localhost")
    port = int(os.environ.get("UVICORN_PORT", "8000"))
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
