import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MultiLinearRegression(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


def load_bundle(bundle_path: str):
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    # Bundle must contain these keys
    required_keys = ["input_size", "feature_columns", "scaler_X", "scaler_Y", "state_dict"]
    for k in required_keys:
        if k not in bundle:
            raise KeyError(f"model_bundle.pkl missing key: '{k}'")

    input_size = bundle["input_size"]
    feature_columns = bundle["feature_columns"]
    scaler_X = bundle["scaler_X"]
    scaler_Y = bundle["scaler_Y"]
    state_dict = bundle["state_dict"]

    model = MultiLinearRegression(input_size)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler_X, scaler_Y, feature_columns


def predict_from_dict(model, scaler_X, scaler_Y, feature_columns, user_input: dict):
    # Create empty row with all features
    row = {col: 0 for col in feature_columns}

    # Fill values from user input
    for k, v in user_input.items():
        if k in row:
            row[k] = v

    df_input = pd.DataFrame([row])

    # Scale input with X scaler
    X_scaled = scaler_X.transform(df_input.values)

    # Predict (returns scaled prediction)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(x_tensor).numpy().reshape(1, -1)

    # CRITICAL: Inverse transform to original units using Y scaler
    pred_original = scaler_Y.inverse_transform(pred_scaled)[0][0]

    return float(pred_original)

