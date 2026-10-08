# RailKavach Edge AI

RailKavach Edge AI is a hackathon-ready software prototype for decentralized railway track monitoring and pre-collision hazard intervention in remote corridors where cellular connectivity may be unavailable.

The project models ruggedized trackside edge nodes placed at fixed intervals. Each node combines three local sensing layers:

1. Structural health sensing using vibration, acceleration, rail temperature, gauge deviation, and ballast-settlement signals
2. Geometric clearance monitoring using LiDAR-style minimum-distance measurements
3. Edge vision using YOLO for track-intrusion detection

A local fusion engine combines the sensor risks and generates a hazard level. When risk is high, the prototype creates a compact V2I packet and relays it through a simulated Sub-GHz LoRa mesh to a locomotive-side Driver Machine Interface simulator. The DMI calculates time-to-hazard, estimated stopping distance, distance margin, and a recommended response.

## Important Safety Scope

This repository is a hackathon prototype only.

It does not connect to real railway signalling, TCMS, automatic train protection, braking systems, LoRa hardware, safety PLCs, or certified sensors. The "emergency brake command" shown in the interface is only a simulated software decision.

No part of this code should be used for real train control without railway-authority approval, fail-safe engineering, certified hardware, independent verification, and compliance with applicable railway signalling and functional-safety standards.

## Main Features

- Decentralized trackside edge-node simulation
- Structural-health machine-learning model
- Rail-fracture pattern detection
- Fastener-displacement detection
- Track-misalignment detection
- Ballast-degradation detection
- LiDAR-style protected-clearance monitoring
- Optional YOLO object / wildlife intrusion detection
- Multi-modal hazard fusion
- LOW / MODERATE / HIGH / CRITICAL risk levels
- Local acoustic-deterrent simulation
- Offline Sub-GHz LoRa mesh packet simulation
- Packet integrity tag
- Locomotive DMI simulator
- Distance-to-hazard countdown
- Estimated stopping-distance calculation
- Advisory / braking-response logic
- Event logging and analytics
- Corridor node map
- Synthetic structural-health training dataset
- `.env` configuration

## Problem Statement

India operates one of the world's most extensive rail networks, and large sections of track are continuously exposed to thermal stress, ballast deterioration, structural fatigue, severe weather, wildlife movement, landslides, debris, and vehicle intrusions.

Routine inspection often depends on manual patrols and periodic inspection vehicles. This creates monitoring gaps in which rail fractures, displaced fasteners, track misalignment, ballast failure, and unexpected track obstructions can remain undetected.

The challenge becomes more serious in tunnels, mountain cuttings, sharp curves, forest ghats, and fog-prone routes. In these areas, a locomotive crew may not be able to see an obstruction until the train is already too close to stop safely. Cellular and cloud connectivity can also be weak or completely unavailable in remote terrain, making cloud-dependent monitoring unreliable.

A resilient railway-safety architecture therefore needs to detect hazards locally, make decisions at the edge, and transmit warnings to approaching trains without depending on cellular infrastructure.

## Proposed Solution

RailKavach Edge AI demonstrates a decentralized Vehicle-to-Infrastructure edge-computing architecture.

Ruggedized trackside nodes can conceptually be installed at fixed intervals in vulnerable corridors. Each node performs local sensing and AI processing.

### Structural Health Monitoring

The prototype uses:

- RMS rail vibration
- Peak acceleration
- Dominant vibration frequencies
- Rail temperature
- Gauge deviation
- Ballast settlement

A Random Forest classifier predicts:

- Healthy
- Rail Fracture
- Fastener Displacement
- Track Misalignment
- Ballast Degradation

### LiDAR Clearance Monitoring

A simulated LiDAR module measures the nearest intrusion distance to the protected rail envelope.

If the distance falls below configured thresholds, the node raises an escalating clearance risk.

### Edge Vision

An optional YOLO model analyzes uploaded track images and searches for configured intrusion classes such as:

- People
- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles
- Cattle / animals

The first YOLO run may download the configured model if it is not already installed.

### Sensor Fusion

Structural, LiDAR, and vision risk values are fused into a single hazard score.

A critical result from one sensor can dominate the fused score so that an obvious obstruction is not hidden by otherwise healthy track data.

### Local Response

For HIGH or CRITICAL risk, the prototype simulates:

- Local acoustic deterrent activation
- Hazard packet generation
- Offline mesh transmission
- Locomotive warning / braking recommendation

### Offline V2I Mesh

The prototype creates a compact hazard packet containing:

- Trackside node ID
- Chainage
- Risk score
- Risk level
- Hazard type
- Distance to approaching train
- Timestamp
- Integrity tag

The packet is then forwarded through a simulated LoRa/Sub-GHz relay path.

No real radio transmission is performed.

### Locomotive DMI

The locomotive-side simulator calculates:

- Current distance to hazard
- Time to hazard
- Estimated stopping distance
- Distance margin
- Recommended driver / braking response

The displayed emergency-brake command is a software simulation only.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Joblib
- OpenCV
- Ultralytics YOLO
- Pillow
- python-dotenv

## Project Structure

```text
railkavach_edge_ai/
├── app.py
├── requirements.txt
├── .env
├── .env.example
├── README.md
├── HOW_IBM_BOB_WAS_USED.md
├── run_app.bat
├── data/
│   ├── structural_sensor_data.csv
│   └── trackside_nodes.csv
├── models/
│   ├── structural_model.joblib
│   └── structural_metrics.json
├── utils/
│   ├── structural.py
│   ├── fusion.py
│   ├── braking.py
│   ├── lora.py
│   ├── vision.py
│   └── events.py
├── outputs/
└── .streamlit/
    └── config.toml
```

## Installation

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Windows users can also run:

```text
run_app.bat
```

## Suggested Hackathon Demo

1. Open Command Dashboard and explain the six fixed trackside nodes.
2. Open Trackside Edge Node.
3. Start with normal structural readings, safe LiDAR clearance, and no visual intrusion.
4. Run sensor fusion and show LOW risk.
5. Increase vibration, peak acceleration, and gauge deviation to simulate a structural fault.
6. Reduce LiDAR clearance below the configured threshold.
7. Select an animal, person, or stranded vehicle as the simulated vision intrusion.
8. Run fusion again and show HIGH / CRITICAL risk.
9. Demonstrate local acoustic-deterrent activation.
10. Show the generated V2I hazard packet.
11. Open the LoRa V2I Mesh page and demonstrate multi-hop relay.
12. Open Locomotive DMI Simulator.
13. Change train speed and distance to show how stopping-distance margin changes.
14. Upload a track image to the Vision Intrusion Detection page.
15. Open Event Analytics to show logged hazards.

## Hardware Expansion

A physical prototype could later use:

- Industrial piezoelectric geophone
- Triaxial MEMS accelerometer
- Rail-temperature sensor
- Laser / solid-state LiDAR
- Thermal camera
- Industrial optical camera
- Edge-AI computer
- Sub-GHz LoRa radio
- Solar panel
- LiFePO4 battery
- Rugged IP67 enclosure
- Directional acoustic siren
- Train-side receiver
- Isolated TCMS gateway

Real train-control integration must remain isolated from this prototype until safety certification and railway-authority approval.

## Future Improvements

- Real geophone waveform processing
- FFT and spectral anomaly detection
- Federated / edge model updates
- Thermal-camera fusion
- Depth-aware vision
- True LoRa radio hardware
- Mesh routing and packet-loss simulation
- Node battery / solar-energy management
- Sensor self-diagnostics
- Redundant dual-node confirmation
- Track-circuit / axle-counter integration
- GIS corridor planning
- Digital twin
- Maintenance ticket generation
- Railway control-centre integration
- Hardware-in-the-loop test bench
