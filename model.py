import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv", encoding='latin1')

# Remove unwanted column
if "laptop_ID" in df.columns:
    df = df.drop("laptop_ID", axis=1)

# ---------------- CLEAN DATA ----------------
df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).astype(float)

# ---------------- SAVE VALID VALUES ----------------
valid_companies = df["Company"].unique()
valid_cpus = df["Cpu"].unique()
valid_weights = df["Weight"].unique()

# ---------------- SELECT FEATURES ----------------
df = df[["Company", "Cpu", "Weight", "Price_euros"]]

# Convert categorical to numeric
df = pd.get_dummies(df)

# ---------------- TRAIN MODEL ----------------
X = df.drop("Price_euros", axis=1)
y = df["Price_euros"]

model = RandomForestRegressor()
model.fit(X, y)

# ---------------- SAVE FILES ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))
pickle.dump(valid_companies, open("companies.pkl", "wb"))
pickle.dump(valid_cpus, open("cpus.pkl", "wb"))
pickle.dump(valid_weights, open("weights.pkl", "wb"))

print("✅ Model + all validation files saved successfully!")