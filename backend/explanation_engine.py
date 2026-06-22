def explain_prediction(
    inputs: dict, 
    prediction_class: str, 
    contributions: list, 
    confidence: float, 
    second_confidence: float = 0.0
) -> dict:
    """
    Analyzes model prediction, confidence scores, and SHAP-based feature contributions
    to generate structured, instance-level explanations.
    
    Args:
        inputs (dict): Raw input features selected by the user.
        prediction_class (str): Predicted class (e.g. 'unacc', 'acc', 'good', 'vgood').
        contributions (list): List of dicts representing feature impact scores:
                             [{'feature': '...', 'influence': float}]
        confidence (float): Calculated model confidence probability percentage.
        second_confidence (float): Second highest class probability percentage.
        
    Returns:
        dict: Structured explanation containing summary text, top features,
              decision strength, and confidence reason.
    """
    # Determine user-friendly prediction label
    class_labels = {
        "unacc": "Unacceptable",
        "acc": "Acceptable",
        "good": "Good",
        "vgood": "Very Good"
    }
    pred_label = class_labels.get(prediction_class, prediction_class.capitalize())

    # Pre-defined display mappings for default features and categories
    feature_display_names = {
        "buying": "Buying Price",
        "maint": "Maintenance Cost",
        "doors": "Doors Count",
        "persons": "Persons Capacity",
        "lug_boot": "Luggage Boot Size",
        "safety": "Safety Rating"
    }
    
    value_display_maps = {
        "buying": {"vhigh": "Very High", "high": "High", "med": "Medium", "low": "Low"},
        "maint": {"vhigh": "Very High", "high": "High", "med": "Medium", "low": "Low"},
        "doors": {"2": "2 Doors", "3": "3 Doors", "4": "4 Doors", "5more": "5 or More"},
        "persons": {"2": "2 Persons", "4": "4 Persons", "more": "More than 4"},
        "lug_boot": {"small": "Small", "med": "Medium", "big": "Big"},
        "safety": {"low": "Low Safety", "med": "Medium Safety", "high": "High Safety"}
    }

    pos_features = []
    neg_features = []
    
    # Process contributions dynamically using features from metadata
    for item in contributions:
        feat = item["feature"]
        influence = item["influence"]
        raw_val = inputs.get(feat, "")
        
        # Resolve display names and value formatting dynamically
        display_name = feature_display_names.get(feat, feat.replace("_", " ").title())
        val_map = value_display_maps.get(feat, {})
        value_display = val_map.get(str(raw_val).lower(), str(raw_val).capitalize())
        
        feat_info = {
            "feature": feat,
            "influence": influence,
            "value": value_display,
            "display_name": display_name
        }
        
        if influence > 0:
            pos_features.append(feat_info)
        elif influence < 0:
            neg_features.append(feat_info)
            
    # Sort positive features descending (strongest positive influence first)
    pos_features.sort(key=lambda x: x["influence"], reverse=True)
    
    # Sort negative features ascending (most negative influence first)
    neg_features.sort(key=lambda x: x["influence"])

    # Generate natural language summary using "influence/impact score" terminology
    summary_parts = []
    if pos_features:
        top_pos = pos_features[0]
        summary_parts.append(
            f"The vehicle is evaluated as {pred_label} primarily because of its {top_pos['display_name']} ({top_pos['value']}), "
            f"which contributed a positive influence score of +{top_pos['influence']}."
        )
        if len(pos_features) > 1:
            sec_pos = pos_features[1]
            summary_parts.append(
                f"Additionally, its {sec_pos['display_name']} ({sec_pos['value']}) had a positive influence score of +{sec_pos['influence']}."
            )
    else:
        summary_parts.append(f"The vehicle parameters combined to result in a {pred_label} evaluation.")
            
    summary = " ".join(summary_parts)

    # Compute decision strength based on confidence thresholds
    if confidence >= 90.0:
        decision_strength = "Strong"
    elif confidence >= 75.0:
        decision_strength = "Moderate"
    else:
        decision_strength = "Weak"

    # Compute confidence reason comparing predicted class and second highest probability
    margin = confidence - second_confidence
    if prediction_class == "unacc":
        confidence_reason = (
            f"The model's classification is {decision_strength.lower()} ({confidence:.1f}% confidence) because key attributes "
            f"failed to meet basic acceptability thresholds, leading to a clear margin of {margin:.1f} points."
        )
    else:
        confidence_reason = (
            f"The model has a {decision_strength.lower()} degree of certainty ({confidence:.1f}% confidence) for this evaluation, "
            f"exhibiting an influence margin of {margin:.1f} points over the next alternative classification."
        )

    return {
        "summary": summary,
        "top_positive_features": pos_features,
        "top_negative_features": neg_features,
        "confidence_reason": confidence_reason,
        "decision_strength": decision_strength
    }

