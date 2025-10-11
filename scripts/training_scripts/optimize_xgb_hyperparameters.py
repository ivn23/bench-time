def objective(trial):
    # Focus on the most impactful hyperparameters (80/20 approach)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'random_state': 42
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []

    for train_idx, val_idx in tscv.split(X_t):
        # Split the data into training and validation sets
        X_train = X_t[train_idx].sort("date","skuID").select(FEATURE_COLUMNS).to_numpy()
        y_train = y_t[train_idx].sort("date","skuID").select(TARGET_COLUMN).to_numpy()
        X_val = X_t[val_idx].sort("date","skuID").select(FEATURE_COLUMNS).to_numpy()
        y_val = y_t[val_idx].sort("date","skuID").select(TARGET_COLUMN).to_numpy()

        # Train the model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_val)
        y_pred = y_pred.round().astype(int)

        # Evaluate the model
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)

    # Return the average MSE across all folds
    return np.mean(mse_scores)

# Create study and optimize
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 50)  # 50 trials for good balance of speed/performance

# Get best parameters
best_params = study.best_params
print(f"\nBest parameters found:")
for key, value in best_params.items():
    print(f"{key}: {value}")

# Train final model with best parameters
print(f"\nBest MSE: {study.best_value:.4f}")