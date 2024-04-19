from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')
@app.route('/predict', methods = ['POST', "GET"])

def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data = CustomData(
            Organization_Group_Code = int(request.form.get('Organization_Group_Code')),
            Job_Family_Code = str(request.form.get('Job_Family_Code')),
            Job_Code = str(request.form.get('Job_Code')),
            Year = int(request.form.get('Year')),
            Department_Code = str(request.form.get('Department_Code')),
            Union_Code = float(request.form.get('Union_Code')),
            Overtime_Amount = str(request.form.get('Overtime_Amount')),
            Retirement_Amount = str(request.form.get('Retirement_Amount')),
            Health_and_Dental_Amount = str(request.form.get('Health_and_Dental_Amount')),
            Other_Benefits_Amount = str(request.form.get('Other_Benefits_Amount'))
        )
    new_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data)

    results = round(pred[0],2)

    return render_template("results.html", final_result = results)

if __name__ == "__main__": 
    app.run(host = "127.0.0.1", debug= True)