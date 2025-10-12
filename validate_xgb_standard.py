"""
Validation script for XGBoost Standard refactoring.
Tests that the refactored xgboost_standard model works correctly with the new native API.
"""

import numpy as np
from src.models.xgboost_standard import XGBoostStandardModel

print("=" * 60)
print("XGBoost Standard Model Validation")
print("=" * 60)

# Test 1: Instantiation with seed parameter
print("\n1. Testing instantiation with seed parameter...")
try:
    model = XGBoostStandardModel(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        seed=42
    )
    print("✓ Model instantiated successfully with seed parameter")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 2: Training with native API
print("\n2. Testing training with native XGBoost API...")
try:
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)

    model.train(X_train, y_train)
    print("✓ Model trained successfully using native xgb.train()")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 3: Prediction
print("\n3. Testing prediction...")
try:
    X_test = np.random.rand(20, 10)
    predictions = model.predict(X_test)
    assert len(predictions) == 20
    print(f"✓ Predictions generated successfully (shape: {predictions.shape})")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 4: Model info
print("\n4. Testing model info...")
try:
    info = model.get_model_info()
    assert info["model_class"] == "XGBBooster"
    assert info["training_method"] == "xgb_train"
    assert "seed" in info["parameters"]
    assert info["parameters"]["seed"] == 42
    print("✓ Model info correct:")
    print(f"  - Model class: {info['model_class']}")
    print(f"  - Training method: {info['training_method']}")
    print(f"  - Seed parameter: {info['parameters']['seed']}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 5: No random_state in parameters
print("\n5. Testing that random_state is not in parameters...")
try:
    model2 = XGBoostStandardModel(
        n_estimators=10,
        max_depth=3,
        seed=123
    )
    info = model2.get_model_info()
    assert "random_state" not in info["parameters"]
    assert "seed" in info["parameters"]
    print("✓ Confirmed: seed parameter used (not random_state)")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 6: Verify native XGBoost Booster object
print("\n6. Testing that model uses native XGBoost Booster...")
try:
    import xgboost as xgb
    assert isinstance(model.model, xgb.Booster)
    print(f"✓ Confirmed: model is xgb.Booster (type: {type(model.model).__name__})")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

print("\n" + "=" * 60)
print("ALL VALIDATION TESTS PASSED ✓")
print("=" * 60)
print("\nRefactoring Summary:")
print("  - XGBoostStandardModel now uses native xgb.train() API")
print("  - Uses 'seed' parameter instead of 'random_state'")
print("  - Returns xgb.Booster objects instead of sklearn XGBRegressor")
print("  - All functionality working correctly")
