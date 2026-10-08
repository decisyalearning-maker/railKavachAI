def structural_risk(row):
    score = 0
    fault = str(row.get("predicted_fault","Healthy"))

    if fault == "Rail Fracture":
        score += 75
    elif fault == "Track Misalignment":
        score += 60
    elif fault == "Fastener Displacement":
        score += 50
    elif fault == "Ballast Degradation":
        score += 40

    if float(row.get("gauge_deviation_mm",0)) >= 5:
        score += 20
    elif float(row.get("gauge_deviation_mm",0)) >= 3:
        score += 10

    if float(row.get("peak_acceleration_g",0)) >= 8:
        score += 15
    elif float(row.get("peak_acceleration_g",0)) >= 6:
        score += 8

    if float(row.get("ballast_settlement_mm",0)) >= 15:
        score += 15
    elif float(row.get("ballast_settlement_mm",0)) >= 10:
        score += 8

    return min(score, 100)

def lidar_risk(clearance_m, threshold_m=1.8):
    clearance_m = float(clearance_m)
    if clearance_m <= 0.6:
        return 100
    if clearance_m <= 1.0:
        return 90
    if clearance_m <= threshold_m:
        return 70
    if clearance_m <= threshold_m + 0.7:
        return 35
    return 0

def vision_risk(detections):
    if not detections:
        return 0
    weights = {
        "person": 90,
        "car": 100,
        "truck": 100,
        "bus": 100,
        "motorcycle": 90,
        "bicycle": 80,
        "cow": 90,
        "horse": 90,
        "sheep": 80,
        "dog": 65,
        "elephant": 100,
        "bear": 100
    }
    score = 0
    for d in detections:
        score = max(score, weights.get(str(d.get("label","")).lower(), 55))
    return min(score, 100)

def fused_risk(structural_score, lidar_score, vision_score):
    # Max-sensitive fusion: a single verified critical hazard should dominate.
    weighted = (
        float(structural_score) * 0.45
        + float(lidar_score) * 0.30
        + float(vision_score) * 0.25
    )
    dominant = max(structural_score, lidar_score, vision_score)
    if dominant >= 90:
        weighted = max(weighted, 85)
    elif dominant >= 75:
        weighted = max(weighted, 72)
    return round(min(weighted, 100), 1)

def risk_level(score):
    if score >= 85:
        return "CRITICAL"
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MODERATE"
    return "LOW"
