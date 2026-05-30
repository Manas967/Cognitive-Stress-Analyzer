import os
import pickle
import pandas as pd #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore
from sklearn.ensemble import RandomForestClassifier #type:ignore

def train_and_save_pipeline():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    df = pd.read_csv("stress_data.csv")
    
    X = df[['heart_rate', 'work_duration_hours', 'environment_noise', 'sleep_hours']]
    y = df['stress_level']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    with open(os.path.join(model_dir, "stress_model.pkl"), "wb") as m_file:
        pickle.dump(model, m_file)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as s_file:
        pickle.dump(scaler, s_file)

if __name__ == "__main__":
    train_and_save_pipeline()