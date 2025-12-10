from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page

@app.route('/')
def index():
    df = pd.read_csv('artifacts/test.csv')  
    users = df['user_id'].unique()
    products = df['product_id'].unique()
    return render_template('home.html', users=users, products=products)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # Read the CSV file to get user and product lists for both GET and POST
    df = pd.read_csv('artifacts/test.csv')  
    users = df['user_id'].unique()
    products = df['product_id'].unique()
    
    if request.method == 'GET':
        return render_template('home.html', users=users, products=products)
    else:
        data = CustomData(
            user_id=request.form.get('user_id'),
            product_id=request.form.get('product_id')
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', 
                             results=results[0][0], 
                             users=users, 
                             products=products)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)