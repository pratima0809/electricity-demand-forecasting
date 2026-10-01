def predict_next_hour(model, features, current_features_df):
    """Predict the next hour given the engineered features DataFrame row."""
    X = current_features_df[features]
    return model.predict(X)[0]
