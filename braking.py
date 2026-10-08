def kmph_to_mps(speed_kmph):
    return float(speed_kmph) / 3.6

def braking_distance_m(speed_kmph, deceleration_mps2=0.65, reaction_buffer_s=5):
    v = kmph_to_mps(speed_kmph)
    reaction_distance = v * float(reaction_buffer_s)
    braking_distance = (v * v) / (2 * float(deceleration_mps2))
    return round(reaction_distance + braking_distance, 1)

def time_to_hazard_s(distance_m, speed_kmph):
    v = kmph_to_mps(speed_kmph)
    if v <= 0:
        return None
    return round(float(distance_m) / v, 1)

def intervention_decision(distance_m, speed_kmph, risk_level, deceleration_mps2=0.65, reaction_buffer_s=5):
    stop_distance = braking_distance_m(speed_kmph, deceleration_mps2, reaction_buffer_s)
    tth = time_to_hazard_s(distance_m, speed_kmph)

    if risk_level == "CRITICAL":
        if distance_m <= stop_distance * 1.15:
            action = "EMERGENCY BRAKE COMMAND"
        else:
            action = "PRE-EMPTIVE BRAKING / DRIVER ACKNOWLEDGEMENT"
    elif risk_level == "HIGH":
        if distance_m <= stop_distance:
            action = "EMERGENCY BRAKE COMMAND"
        else:
            action = "REDUCE SPEED IMMEDIATELY"
    elif risk_level == "MODERATE":
        action = "CAUTION / PREPARE TO BRAKE"
    else:
        action = "MONITOR"

    return {
        "stopping_distance_m": stop_distance,
        "time_to_hazard_s": tth,
        "action": action,
        "margin_m": round(float(distance_m) - stop_distance, 1)
    }
