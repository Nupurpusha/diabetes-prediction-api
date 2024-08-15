from fastapi import FastAPI, Path, Query, Body, Form, File, UploadFile
from pydantic import BaseModel
import pickle
import numpy as np
import sklearn
from typing import Optional

app = FastAPI()

# Define a model
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# Example of a GET request endpoint
@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_igcvggvvgd: int = Path(..., title="The ID of the item to get"), q: Optional[str] = None):
    """
    GET endpoint to retrieve an item by its ID.
    - item_id: The ID of the item (path parameter)
    - q: An optional query parameter
    """
    item = {"name": "Item", "description": "An item", "price": 10.0, "tax": 1}
    if q:
        item.update({"q": q})
    return item

# Example of a POST request endpoint
@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    """
    POST endpoint to create a new item.
    - item: The item data (body parameter)
    """
    return item

# Example of a PUT request endpoint
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: Item):
    """
    PUT endpoint to update an existing item by its ID.
    - item_id: The ID of the item to update (path parameter)
    - item: The updated item data (body parameter)
    """
    updated_item = item.dict()
    updated_item.update({"id": item_id})
    return updated_item

# Example of a DELETE request endpoint
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """
    DELETE endpoint to delete an item by its ID.
    - item_id: The ID of the item to delete (path parameter)
    """
    return {"message": f"Item {item_id} deleted"}

# Example of a PATCH request endpoint (partial update)
@app.patch("/items/{item_id}", response_model=Item)
async def patch_item(item_id: int, item: Item):
    """
    PATCH endpoint to partially update an existing item by its ID.
    - item_id: The ID of the item to update (path parameter)
    - item: The partial item data (body parameter)
    """
    patched_item = item.dict(exclude_unset=True)
    patched_item.update({"id": item_id})
    return patched_item

# Example of handling form data with a POST request
@app.post("/form/")
async def handle_form_data(name: str = Form(...), age: int = Form(...)):
    """
    POST endpoint to handle form data.
    - name: Name from the form (form data)
    - age: Age from the form (form data)
    """
    return {"name": name, "age": age}

# Example of handling file uploads
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """
    POST endpoint to handle file uploads.
    - file: The uploaded file (file data)
    """
    return {"filename": file.filename}

# Example of handling multiple query parameters
@app.get("/search/")
async def search_items(query: str, limit: int = 10, offset: int = 0):
    """
    GET endpoint to search items with multiple query parameters.
    - query: The search query (query parameter)
    - limit: The number of results to return (query parameter)
    - offset: The starting position of the results (query parameter)
    """
    return {"query": query, "limit": limit, "offset": offset, "results": ["item1", "item2"]}

# To run the FastAPI app, use the command: uvicorn script_name:app --reload

@app.post("/predict_diabetes/")
async def predict_diabetes(Pregnancies:int, Glucose:int, BloodPressure:int, SkinThickness:int, Insulin:int, BMI:float, DiabetesPedigreeFunction:float, Age:int):
    """
    Predict diabetes based on input features using a pre-trained Random Forest classifier.

    Parameters:
    Pregnancies (int): Number of pregnancies
    Glucose (float): Glucose level
    BloodPressure (float): Blood pressure level
    SkinThickness (float): Skin thickness
    Insulin (float): Insulin level
    BMI (float): Body Mass Index
    DiabetesPedigreeFunction (float): Diabetes pedigree function
    Age (int): Age in years
    model_path (str): Path to the pickle file containing the trained model

    Returns:
    int: Predicted class (0 or 1)
    """
    # Load the Random Forest classifier from the pickle file
    with open('/Users/aadilgarg/Downloads/TSS-FastAPI/ThaparSummerSchool2024/FastAPI/diabetes_model.pkl', 'rb') as file:
        rf_classifier = pickle.load(file)
    
    # Create a numpy array from the input features
    input_features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Make a prediction using the Random Forest classifier
    prediction = rf_classifier.predict(input_features)

    return int(prediction[0])
