def explain_prediction(inputs: dict, prediction_class: str) -> str:
    """
    Analyzes input features and predicted class to generate a dynamic,
    instance-level explanation text of why the car received its classification.
    """
    buying = inputs.get("buying", "").lower()
    maint = inputs.get("maint", "").lower()
    doors = inputs.get("doors", "")
    persons = inputs.get("persons", "").lower()
    lug_boot = inputs.get("lug_boot", "").lower()
    safety = inputs.get("safety", "").lower()

    # 1. Unacceptable (unacc) dealbreakers
    if prediction_class == "unacc":
        if safety == "low":
            return (
                "This vehicle is classified as Unacceptable primarily due to its Low safety rating. "
                "Safety is the most critical feature in our evaluation model; any vehicle with low safety "
                "is automatically rejected, regardless of any other positive characteristics."
            )
        if persons == "2":
            return (
                "This vehicle is classified as Unacceptable because it only has a capacity of 2 persons. "
                "Seating capacity is a high-priority feature, and a capacity of 2 is considered insufficient for "
                "acceptable passenger utility, regardless of price or safety."
            )
        if buying == "vhigh" and maint == "vhigh":
            return (
                "This vehicle is unacceptable because of the combination of a Very High buying price and Very High maintenance cost. "
                "The model evaluates this price-to-maintenance ratio as too expensive to justify acceptable vehicle status."
            )
        if buying == "vhigh" or maint == "vhigh":
            cost_type = "buying price" if buying == "vhigh" else "maintenance cost"
            return (
                f"This vehicle is unacceptable because of its Very High {cost_type}. "
                "Even with adequate safety and capacity, a very high cost in either category exceeds acceptable thresholds in the model."
            )
        # Fallback for unacc
        return (
            "This vehicle is evaluated as Unacceptable due to a combination of weak parameters, "
            f"such as its {safety} safety rating, {lug_boot} luggage space, or maintenance costs."
        )

    # 2. Acceptable (acc)
    elif prediction_class == "acc":
        reasons = []
        if safety == "high":
            reasons.append("High safety rating")
        elif safety == "med":
            reasons.append("Medium safety rating")
            
        if persons in ["4", "more"]:
            reasons.append(f"adequate capacity ({persons} persons)")
            
        if buying in ["low", "med"] and maint in ["low", "med"]:
            reasons.append("affordable purchase and maintenance costs")

        reason_str = ", ".join(reasons)
        return (
            f"This vehicle is rated Acceptable because it offers {reason_str}. "
            "However, its rating is held back from 'Good' due to moderate parameters in other categories (such as price or luggage capacity)."
        )

    # 3. Good (good)
    elif prediction_class == "good":
        return (
            f"This vehicle is rated Good because it has a strong combination of High safety, "
            f"sufficient seating capacity ({persons} persons), and very reasonable costs (Buying: {buying.upper()}, Maintenance: {maint.upper()})."
        )

    # 4. Very Good (vgood)
    elif prediction_class == "vgood":
        return (
            f"This vehicle is rated Very Good because it represents an optimal configuration: "
            f"High safety, large passenger capacity ({persons} persons), ample luggage space ({lug_boot.upper()} boot), "
            f"and affordable cost structures (Buying: {buying.upper()}, Maint: {maint.upper()})."
        )

    return "Vehicle parameters evaluated successfully by the Random Forest model."
