# Configuration settings for the project
# All paths and hyperparameters are defined here

import os

# ============================================================================
# PATHS
# ============================================================================

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'creditcard.csv')

# Results paths
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
TEST_EXAMPLES_DIR = os.path.join(RESULTS_DIR, 'test_examples')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')

# Plots paths
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Old outputs (for migration)
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
OLD_MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
OLD_PLOTS_EDA_DIR = os.path.join(OUTPUTS_DIR, 'plots_eda')
OLD_PLOTS_RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'plots_results')

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

# Data split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 3

# SMOTE settings
SMOTE_RANDOM_STATE = 42

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# XGBoost hyperparameter search space
PARAM_GRID = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__scale_pos_weight': [1, 10]
}

# RandomizedSearchCV settings
N_ITER = 5
SCORING = 'f1'
N_JOBS = -1

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

# Number of test examples to generate
N_FRAUD_EXAMPLES = 10
N_NORMAL_EXAMPLES = 10

# ============================================================================
# PLOTTING SETTINGS
# ============================================================================

# Figure sizes
FIGSIZE_SMALL = (6, 5)
FIGSIZE_MEDIUM = (8, 6)
FIGSIZE_LARGE = (10, 8)

# DPI for saved figures
DPI = 100
