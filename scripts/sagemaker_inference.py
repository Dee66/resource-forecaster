"""
SageMaker serverless inference entrypoint template.

This file demonstrates the minimal handlers required by SageMaker inference containers:
 - model_fn: load the model from the local filesystem
 - predict_fn: transform input and return predictions
 - input_fn / output_fn: serialize/deserialize requests

Use this as a starting point when building a custom inference container for SageMaker.
"""
import json
import os
from typing import Any


def model_fn(model_dir: str) -> Any:
    """Load model artifacts from model_dir and return a model object.

    Expect the model artifact to be extracted to model_dir by SageMaker. For example,
    if the model tar contains `model.pkl` or `model.joblib`, load it here.
    """
    import joblib

    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    return model


def input_fn(serialized_input_data: bytes, content_type: str = "application/json") -> Any:
    if content_type == "application/json":
        return json.loads(serialized_input_data.decode("utf-8"))
    raise ValueError("Unsupported content type: %s" % content_type)


def predict_fn(input_data: dict, model: Any) -> dict:
    # Input: a dict matching the inference contract, e.g. {start_date, end_date, granularity, features...}
    # Transform input_data to model features here.
    X = []  # TODO: convert input_data to model features
    preds = model.predict(X)
    return {"predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds)}


def output_fn(prediction: dict, accept: str = "application/json") -> bytes:
    if accept == "application/json":
        return json.dumps(prediction).encode("utf-8")
    raise ValueError("Unsupported accept type: %s" % accept)
