"""
This is a FastAPI application that receives an HTTP GET request at the root
endpoint ("/") and returns the classification probability of a digit image.
"""

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from Model.predict import classify_digit

# Create FastAPI instance
app = FastAPI()


# Define GET route to classify the digit

@app.get("/")
async def get_classify_digit(info: Request):
    # Parse input image from the request
    req_info = await info.json()
    img = np.array(req_info["image"])

    # Classify the digit using the imported function
    prob = classify_digit(img)

    # Return probability of each class in a JSON format
    return {"prob": prob.tolist()}
    #print('Hello')
    #return {"prob": [0.1, 0.5, 0.3,0.5,0.7,0.8,0.9,0.1,0.1,0.1]}
if __name__ == '__main__':
    uvicorn.run(app)