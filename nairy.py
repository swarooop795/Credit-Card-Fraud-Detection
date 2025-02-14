from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)
# Global variables for the model and scaler
model = None
scaler = None

# HTML Templates
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Fraud Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-image: url('https://d1m75rqqgidzqn.cloudfront.net/wp-data/2021/09/22153728/iStock-1203763961.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            color: #333;
        }
        .card {
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent background */
        }
        .card-header {
            background-color: rgba(12, 201, 248, 0.9); /* Slight transparency */
            color: white;
            padding: 20px;
        }
        .btn-start {
            background-color: rgb(23, 43, 4);
            color: white;
            font-size: 22px;
            width: 100%;
        }
        .img-custom {
            max-height: 300px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <div class="card shadow-lg">
            <div class="card-header text-center">
                <h2>Welcome to Credit Card Fraud Detection</h2>
            </div>
            <div class="card-body text-center">
                <img src="https://static1.makeuseofimages.com/wordpress/wp-content/uploads/2015/12/credit-card-fraud.jpg" alt="Credit Card Fraud" class="img-fluid img-custom">
                <form action="/start" method="get">
                    <button type="submit" class="btn btn-start">Start Now</button>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
"""

UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload CSV</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-image: url('https://www.threatx.com/wp-content/uploads/2021/08/Credit%20Card%20Fraud-scaled.jpg');
            background-size: cover;
            background-repeat: no-repeat;
        }
        .card {
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent background */
        }
        .card-header {
            background-color: rgba(143, 235, 23, 0.9); /* Slight transparency */
            color: white;
            font-size: 24px;
        }
        .btn-custom {
            background-color: rgb(153, 46, 224);
            color: white;
            border-radius: 30px;
            width: 100%;
        }
        .btn-home {
            background-color: rgb(107, 110, 112);
            color: white;
            border-radius: 30px;
            width: 100%;
        }
        .img-custom {
            max-height: 300px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <div class="card shadow-lg">
            <div class="card-header text-center">
                <h2>Credit Card Fraud Detection</h2>
            </div>
            <div class="card-body text-center">
                <img src="https://www.bitdefender.com/media/uploads/2018/08/credit-card-fraud.jpeg" alt="Credit Card Fraud" class="img-fluid img-custom">
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <div class="mb-4">
                        <input type="file" class="form-control" name="predict_file" accept=".csv" required>
                    </div>
                    <button type="submit" class="btn btn-custom">Upload and Predict</button>
                </form>
                <form action="/" method="get" class="mt-3">
                    <button type="submit" class="btn btn-home">Go to Home</button>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-image: url('https://www.usnews.com/object/image/0000016b-5271-d190-adeb-5b7b24280000/190613-creditcardfraud-stock.jpg?update-time=1560456466388&size=responsive640');
            background-size: cover;
            background-repeat: no-repeat;
            color: #333;
        }
        .card {
            margin-top: 50px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent background */
        }
        .card-header {
            background-color: rgba(235, 26, 130, 0.9); /* Slight transparency */
            color: white;
        }
        .img-custom {
            max-height: 300px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card shadow-lg">
            <div class="card-header text-center">
                <h2>Transaction Details</h2>
            </div>
            <div class="card-body text-center">
                <img src="https://chargebacks911.com/wp-content/uploads/2023/04/Detecting-Credit-Card-Fraud-blog.jpg" alt="Credit Card Fraud" class="img-fluid img-custom">
                <p><strong>Total Transactions:</strong> {{ results['total_transactions'] }}</p>
                <p><strong>Fraudulent Transactions:</strong> {{ results['fraudulent_cases'] }}</p>
                <p><strong>Fraud Percentage:</strong> {{ results['fraud_percentage'] }}%</p>
                <p><strong>Genuine Transactions:</strong> {{ results['genuine_cases'] }}</p>
            </div>
            <div class="text-center mt-3">
                <form action="/" method="get">
                    <button class="btn btn-secondary">Go Back to Home</button>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/start", methods=["GET"])
def start():
    return render_template_string(UPLOAD_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict_fraud_route():
    global model, scaler

    file = request.files.get("predict_file")
    if not file:
        return render_template_string(UPLOAD_TEMPLATE, results="Please upload a valid CSV file.")

    try:
        data = pd.read_csv(file)
        if 'Class' not in data.columns:
            return render_template_string(UPLOAD_TEMPLATE, results="The file must contain a 'Class' column.")

        X = data.drop('Class', axis=1)
        y = data['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)

        fraudulent_cases = int(report["1"]["support"])
        genuine_cases = int(report["0"]["support"])
        fraud_percentage = (fraudulent_cases / len(y_test)) * 100

        results = {
            "total_transactions": len(y_test),
            "fraudulent_cases": fraudulent_cases,
            "fraud_percentage": round(fraud_percentage, 2),
            "genuine_cases": genuine_cases,
        }

        return render_template_string(RESULTS_TEMPLATE, results=results)

    except Exception as e:
        return render_template_string(UPLOAD_TEMPLATE, results=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
