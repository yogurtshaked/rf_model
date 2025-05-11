@app.post("/predict-nutrient")
def predict_nutrients(data: SensorData) -> Dict:
    # Prepare the input for nutrient prediction
    input_df = pd.DataFrame([{
        'Temperature (°C)': data.temperature,
        'Humidity (%)': data.humidity,
        'TDS Value (ppm)': data.tds,
        'pH Level': data.ph
    }])

    results = {}

    # Predict for each nutrient using the models
    for variable, model in nutrient_model.items():
        pred = model.predict(input_df)[0]
        clean_var = variable.replace(" (°C)", "").replace(" (%)", "").replace(" Value (ppm)", "").replace(" Level", "")
        low, high = normal_ranges[clean_var.lower()]

        # For Temperature and Humidity, no adjustment is needed, only status
        if clean_var in ['Temperature', 'Humidity']:
            status = "Normal" if low <= pred <= high else "Out of Range"
            results[clean_var] = {
                "predicted_value": round(pred, 2),
                "status": status,
                "adjustment": None  # No adjustment for Temperature and Humidity
            }
        else:  # For TDS and pH, check if they are in range
            status = "Normal" if low <= pred <= high else "Out of Range"
            adjustment = None

            # Adjustments for TDS and pH only
            if clean_var in ['TDS Value', 'pH']:
                if pred < low:
                    adjustment = f"Increase by {low - pred:.2f}"
                elif pred > high:
                    adjustment = f"Decrease by {pred - high:.2f}"

            results[clean_var] = {
                "predicted_value": round(pred, 2),
                "status": status,
                "adjustment": adjustment  # Only set if out of range
            }

    return results
