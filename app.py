from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

valid_companies = pickle.load(open("companies.pkl", "rb"))
valid_cpus = pickle.load(open("cpus.pkl", "rb"))
valid_weights = pickle.load(open("weights.pkl", "rb"))

valid_companies_clean = [c.strip().lower() for c in valid_companies]
valid_cpus_clean = [c.strip().lower() for c in valid_cpus]
valid_weights_clean = [round(float(w), 2) for w in valid_weights]

EURO_TO_INR = 90

# ---------------- COMMON STYLE ----------------
def page_style():
    return '''
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: #0f172a;
            color: white;
            text-align: center;
            padding-top: 50px;
            overflow: hidden;
        }
        .container {
            background: #1e293b;
            padding: 30px;
            border-radius: 15px;
            width: 360px;
            margin: auto;
            box-shadow: 0px 0px 25px rgba(0,0,0,0.6);
        }
        input {
            width: 90%;
            padding: 12px;
            margin: 10px 0;
            border-radius: 8px;
            border: none;
            background: #334155;
            color: white;
        }
        button {
            background: #38bdf8;
            color: black;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
        }
        .error {
            color: #f87171;
            font-weight: bold;
        }

        /* 🎉 CONFETTI */
        .confetti {
            position: fixed;
            width: 10px;
            height: 10px;
            background: red;
            bottom: 0;
            animation: blast 2s ease-out forwards;
        }

        @keyframes blast {
            0% { transform: translateY(0) rotate(0); opacity: 1; }
            100% { transform: translateY(-600px) rotate(720deg); opacity: 0; }
        }
    </style>
    '''

# ---------------- HOME ----------------
@app.route('/')
def home():
    return f'''
    <html>
    <head>
        <title>Laptop Price Predictor</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
        {page_style()}
    </head>
    <body>

    <div class="container">
        <h2>💻 Laptop Price Predictor (₹)</h2>

        <form method="POST" action="/predict">
            <input name="Company" placeholder="Company (Dell)" required>
            <input name="Cpu" placeholder="CPU (Intel Core i3)" required>
            <input name="Weight" placeholder="Weight (1.37)" required>
            <br>
            <button type="submit">Predict Price</button>
        </form>
    </div>

    </body>
    </html>
    '''

# ---------------- CPU MATCH ----------------
def find_closest_cpu(user_cpu):
    user_cpu = user_cpu.lower()
    for cpu in valid_cpus_clean:
        if user_cpu in cpu:
            return cpu
    return None

# ---------------- ERROR PAGE ----------------
def error_page(message):
    return f'''
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
        {page_style()}
    </head>
    <body>

    <div class="container">
        <h2 class="error">❌ {message}</h2>
        <br>
        <a href="/">🔙 Try Again</a>
    </div>

    </body>
    </html>
    '''

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    company_input = data.get("Company").strip().lower()
    cpu_input = data.get("Cpu").strip().lower()

    try:
        weight_input = round(float(data.get("Weight")), 2)
    except:
        return error_page("Invalid Weight Format!")

    if company_input not in valid_companies_clean:
        return error_page("Invalid Company Name!")

    matched_cpu = find_closest_cpu(cpu_input)
    if matched_cpu is None:
        return error_page("Invalid CPU! Try i3 / i5 / i7")

    if weight_input not in valid_weights_clean:
        return error_page("Invalid Weight!")

    try:
        original_cpu = valid_cpus[valid_cpus_clean.index(matched_cpu)]
        original_company = valid_companies[valid_companies_clean.index(company_input)]

        input_df = pd.DataFrame([{
            "Company": original_company,
            "Cpu": original_cpu,
            "Weight": weight_input
        }])

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prediction_euro = model.predict(input_df)[0]
        prediction_inr = prediction_euro * EURO_TO_INR

        return f'''
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
            {page_style()}
        </head>
        <body>

        <!-- 🎉 CONFETTI CONTAINER -->
        <script>
            function createConfetti() {{
                for (let i = 0; i < 80; i++) {{
                    let confetti = document.createElement("div");
                    confetti.className = "confetti";

                    // Random colors
                    let colors = ["#f43f5e","#22c55e","#eab308","#38bdf8","#a855f7"];
                    confetti.style.background = colors[Math.floor(Math.random()*colors.length)];

                    // Left or right blast
                    confetti.style.left = (Math.random() < 0.5 ? Math.random()*50 : 50 + Math.random()*50) + "vw";

                    confetti.style.animationDuration = (Math.random()*2 + 1) + "s";

                    document.body.appendChild(confetti);

                    setTimeout(() => confetti.remove(), 3000);
                }}
            }}

            window.onload = createConfetti;
        </script>

        <div class="container">
            <h2>💻 Prediction Result</h2>
            <h1 style="color:#38bdf8;">₹{round(prediction_inr, 2)}</h1>
            <br>
            <a href="/">🔙 Try Again</a>
        </div>

        </body>
        </html>
        '''

    except:
        return error_page("Prediction Error!")

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)