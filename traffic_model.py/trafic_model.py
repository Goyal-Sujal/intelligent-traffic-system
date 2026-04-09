import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 1: Create Dummy Dataset
# -----------------------------
data = {
    "vehicle_count": [10, 20, 50, 80, 120, 30, 60, 90, 150, 200],
    "avg_speed": [60, 55, 40, 30, 20, 50, 35, 25, 15, 10],
    "waiting_time": [5, 10, 20, 30, 50, 15, 25, 35, 60, 80],
    "traffic_level": ["Low", "Low", "Medium", "Medium", "High",
                      "Low", "Medium", "Medium", "High", "High"]
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Prepare Data
# -----------------------------
X = df[["vehicle_count", "avg_speed", "waiting_time"]]
y = df["traffic_level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# Step 3: Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Predict Traffic
# -----------------------------
def predict_traffic(vehicle_count, avg_speed, waiting_time):
    input_data = pd.DataFrame([[vehicle_count, avg_speed, waiting_time]],
                          columns=["vehicle_count", "avg_speed", "waiting_time"])
    prediction = model.predict(input_data)
    return prediction[0]

# -----------------------------
# Step 5: Adaptive Signal Control
# -----------------------------
def traffic_signal_control(traffic_level):
    if traffic_level == "Low":
        return "Green Light: 20 seconds"
    elif traffic_level == "Medium":
        return "Green Light: 40 seconds"
    else:
        return "Green Light: 60 seconds"

# -----------------------------
# Step 6: Test System
# -----------------------------
vehicle_count = int(input("Enter vehicle count: "))
avg_speed = int(input("Enter average speed: "))
waiting_time = int(input("Enter waiting time: "))

traffic = predict_traffic(vehicle_count, avg_speed, waiting_time)
signal = traffic_signal_control(traffic)

print("\nPredicted Traffic Level:", traffic)
print("Signal Adjustment:", signal)