
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('my_api')

# Define predict function
@app.post('/predict')
def predict(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    data = pd.DataFrame([[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
    data.columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)