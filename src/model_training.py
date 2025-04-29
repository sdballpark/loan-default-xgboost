import joblib
import xgboost as xgb
from src.data_preprocessing import preprocess_data, load_data

def train_model(data_path, model_path):
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model('data/raw/loan_data.csv', 'models/loan_default_model.pkl')
