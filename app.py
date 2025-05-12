import os
import dotenv
dotenv.load_dotenv()
from typing import List, Dict
from fastapi import FastAPI, HTTPException
import uvicorn
from model import DBSCANClustering

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/fit")
def fit(data: List[Dict]):
    try:
        model = DBSCANClustering()
        
        # Fit the model

        # Save the model to a file

        return {"message": "Model trained and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(data: List[Dict]):
    try:
        # Load the model from the file

        # Prepare input data for prediction
        
        # Make predictions
        predictions: List = None # Placeholder for actual prediction logic

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__=="__main__":
    uvicorn.run("app:app", host=os.getenv("HOST"), port=int(os.getenv("PORT")), reload=True)