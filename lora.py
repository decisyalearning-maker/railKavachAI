import hashlib
import json
import time

def make_packet(node_id, chainage_km, risk_score, risk_level, hazard_type, distance_to_train_m):
    payload = {
        "node_id": node_id,
        "chainage_km": float(chainage_km),
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "hazard_type": hazard_type,
        "distance_to_train_m": float(distance_to_train_m),
        "timestamp": int(time.time())
    }
    body = json.dumps(payload, sort_keys=True)
    payload["integrity_tag"] = hashlib.sha256(body.encode()).hexdigest()[:16]
    return payload

def relay_packet(packet, relay_nodes=3):
    hops = []
    for i in range(max(1, int(relay_nodes))):
        hops.append({
            "hop": i + 1,
            "relay_id": f"RELAY-{i+1:02d}",
            "status": "FORWARDED"
        })
    return {
        "protocol": "Sub-GHz LoRa Mesh Simulation",
        "packet": packet,
        "hops": hops,
        "delivered": True
    }
