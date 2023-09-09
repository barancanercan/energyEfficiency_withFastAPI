from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the trained model using the correct file path
model = joblib.load("/home/baran/Desktop/PythonProjects/EnergyEfficiency/your_trained_model.joblib")

class InputData(BaseModel):
    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: float
    X7: float
    X8: float
    y1: float
    y2: float

class OutputData(BaseModel):
    prediction: float  # Modify this field to match your model's output

@app.get("/")
def root():
    return {"message": "Welcome to the Energy Efficiency Prediction API!"}

@app.post("/predict-y1/", response_model=OutputData)
def predict_y1(input_data: InputData):
    # Prepare the input data for the model as a dictionary
    input_dict = {
        "X1": input_data.X1,
        "X2": input_data.X2,
        "X3": input_data.X3,
        "X4": input_data.X4,
        "X5": input_data.X5,
        "X6": input_data.X6,
        "X7": input_data.X7,
        "X8": input_data.X8,
    }

    # Make a prediction for y1 using your model
    prediction1 = model.predict([list(input_dict.values())])[0]

    # Create the output data
    output_data1 = {"prediction": prediction1}

    return output_data1

@app.post("/predict-y2/", response_model=OutputData)
def predict_y2(input_data: InputData):
    # Prepare the input data for the model as a dictionary
    input_dict = {
        "X1": input_data.X1,
        "X2": input_data.X2,
        "X3": input_data.X3,
        "X4": input_data.X4,
        "X5": input_data.X5,
        "X6": input_data.X6,
        "X7": input_data.X7,
        "X8": input_data.X8,
    }

    # Make a prediction for y2 using your model
    prediction2 = model.predict([list(input_dict.values())])[1]

    # Create the output data
    output_data2 = {"prediction": prediction2}

    return output_data2
