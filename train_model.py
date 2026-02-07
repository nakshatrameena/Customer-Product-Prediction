import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/customer_data.csv")

# Encode categorical variables
le_gender = LabelEncoder()
le_category = LabelEncoder()
le_target = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["last_product_category"] = le_category.fit_transform(df["last_product_category"])
df["target_product"] = le_target.fit_transform(df["target_product"])

# Features & target
X = df.drop(["customer_id", "target_product"], axis=1)
y = df["target_product"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully.")
