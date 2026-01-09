import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def is_classification_problem(y):
    return (y.nunique() < 20) and (y.dtype in [np.int64, np.int32, object, 'int64', 'int32'])

def run_automl(df, target_col):
    if isinstance(target_col, (list, tuple)) and len(target_col) > 1:
        raise NotImplementedError('Multi-target not implemented.')
    target = target_col if isinstance(target_col, str) else target_col[0]
    X = df.drop(columns=[target])
    y = df[target].copy()
    model_type = 'classification' if is_classification_problem(y) else 'regression'
    X = X.select_dtypes(include=[np.number]).fillna(0)
    if X.shape[1] == 0:
        return {'error': 'No numeric features available for modeling.'}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == 'classification':
        models = {
            'LogisticRegression': (LogisticRegression(max_iter=1000), {'clf__C':[0.1,1]}),
            'RandomForest': (RandomForestClassifier(), {'clf__n_estimators':[100], 'clf__max_depth':[5,10,None]}),
            'GradientBoosting': (GradientBoostingClassifier(), {'clf__n_estimators':[100], 'clf__learning_rate':[0.05,0.1]}),
            'SVC': (SVC(), {'clf__C':[0.1,1], 'clf__kernel':['linear']}),
            'KNN': (KNeighborsClassifier(), {'clf__n_neighbors':[3,5]}),
            'DecisionTree': (DecisionTreeClassifier(), {'clf__max_depth':[5,10,None]})
        }
        scoring = 'accuracy'
    else:
        models = {
            'LinearRegression': (LinearRegression(), {}),
            'RandomForest': (RandomForestRegressor(), {'clf__n_estimators':[100], 'clf__max_depth':[5,10,None]}),
            'GradientBoosting': (GradientBoostingRegressor(), {'clf__n_estimators':[100], 'clf__learning_rate':[0.05,0.1]}),
            'SVR': (SVR(), {'clf__C':[0.1,1], 'clf__kernel':['linear']}),
            'KNN': (KNeighborsRegressor(), {'clf__n_neighbors':[3,5]}),
            'DecisionTree': (DecisionTreeRegressor(), {'clf__max_depth':[5,10,None]})
        }
        scoring = 'r2'
    best_model = None
    best_score = -np.inf
    best_name = None
    results = []
    for name, (model, params) in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        grid = GridSearchCV(pipe, params, cv=3, scoring=scoring, n_jobs=-1)
        try:
            grid.fit(X_train, y_train)
        except Exception as e:
            results.append({'model': name, 'error': str(e)})
            continue
        y_pred = grid.predict(X_test)
        score = accuracy_score(y_test, y_pred) if model_type=='classification' else r2_score(y_test, y_pred)
        results.append({'model': name, 'score': float(score), 'best_params': grid.best_params_})
        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_name = name
    if best_model is not None:
        model_path = os.path.join(MODEL_DIR, f'best_{target}.joblib')
        joblib.dump({'model': best_model, 'features': list(X.columns)}, model_path)
    else:
        model_path = None
    return {'problem_type': model_type, 'best_model_name': best_name, 'best_score': best_score, 'model_path': model_path, 'results': results}
