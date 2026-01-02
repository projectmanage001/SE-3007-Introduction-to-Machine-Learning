import joblib
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def train_model_with_tuning(X_train, y_train, return_search=False):
    
    # We use a Pipeline to ensure SMOTE is only applied during training, 
    # not validation, to prevent data leakage.
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(eval_metric='logloss', use_label_encoder=False))
    ])

    # Task 7: Hyperparameter Tuning
    # We use RandomizedSearchCV because GridSearchCV takes too long for this project scope.
    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__scale_pos_weight': [1, 10] # Helps with imbalance even with SMOTE
    }

    # StratifiedKFold ensures each fold has the same proportion of fraud cases
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("Starting Hyperparameter Tuning (this might take a while)...")
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid, 
        n_iter=5, # Limited iterations for speed in this demo
        scoring='f1', # We optimize for F1-Score, not Accuracy!
        cv=cv, 
        verbose=1, 
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    
    print(f"Best Parameters found: {search.best_params_}")
    print(f"Best F1 Score during validation: {search.best_score_}")

    if return_search:
        return search.best_estimator_, search
    return search.best_estimator_

def save_artifacts(model, scaler, output_dir):
    """
    Task 11: Exporting the pipeline.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    joblib.dump(model, f"{output_dir}/final_model_pipeline.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")
    print("Model and Scaler saved successfully.")