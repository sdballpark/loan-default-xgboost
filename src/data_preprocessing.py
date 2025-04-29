import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data.dropna()
    # One-hot encode 'purpose' and 'term'
    data = pd.get_dummies(data, columns=['purpose', 'term'], drop_first=True)
    
    X = data.drop('default', axis=1)
    y = data['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
