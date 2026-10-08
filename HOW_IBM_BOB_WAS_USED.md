# How IBM Bob Was Used in RailKavach Edge AI

IBM Bob was used as an AI-assisted development tool throughout the RailKavach Edge AI project. It helped break the railway-safety problem into separate engineering layers: structural-health monitoring, geometric clearance sensing, edge vision, local sensor fusion, offline V2I communication, and locomotive-side decision support.

IBM Bob supported the design of the Python and Streamlit project structure and assisted in generating and refining code for the structural-health machine-learning pipeline. It helped organize vibration, acceleration, frequency, temperature, gauge-deviation, and ballast-settlement inputs and supported implementation of the Random Forest classifier used to distinguish Healthy, Rail Fracture, Fastener Displacement, Track Misalignment, and Ballast Degradation patterns.

For the trackside edge layer, IBM Bob assisted in creating the LiDAR clearance logic, vision-intrusion workflow, multi-modal hazard-fusion rules, risk classification, event logging, and corridor-node dashboard. It also helped integrate optional YOLO-based object detection for people, road vehicles, and animal intrusions.

IBM Bob was also used to structure the software representation of the offline V2I communication flow. It helped define the hazard-packet format, packet-integrity tag, simulated multi-hop LoRa mesh relay, and locomotive DMI logic for distance-to-hazard, time-to-hazard, stopping-distance estimation, safety margin, and advisory/braking recommendations.

During development, IBM Bob assisted with debugging, refactoring repeated logic into reusable modules, configuration through `.env`, validation, fallback handling, documentation, and identification of safety limitations.

IBM Bob was used as a development assistant and not as a certified railway-control component. The actual structural prediction in the prototype is performed by a Scikit-learn Random Forest model, while YOLO performs optional visual object detection. All braking outputs, LoRa packets, acoustic warnings, and TCMS/DMI behavior in this repository are simulations. Final architecture choices, testing, safety assumptions, and deployment decisions remain the responsibility of the development team and qualified railway engineers.

## Example IBM Bob Prompts

### Ask Mode

```text
Analyze a railway hazard-monitoring problem where tunnels, mountain cuttings,
forest corridors and poor cellular connectivity make conventional camera and
cloud monitoring unreliable. Identify the sensing, edge-AI and V2I modules
required for a decentralized safety prototype.
```

### Plan Mode

```text
Create a modular Python and Streamlit architecture for a railway trackside
edge-computing prototype. Include structural vibration anomaly detection,
LiDAR clearance monitoring, YOLO-based track intrusion detection, sensor
fusion, LoRa mesh packet simulation, locomotive DMI countdown, stopping-distance
logic, event analytics and offline operation.
```

### Agent Mode

```text
Implement the complete hackathon prototype across multiple Python files.
Create the structural-health dataset and ML model, edge-node simulator,
LiDAR risk logic, YOLO vision module, multi-modal fusion, LoRa packet and
relay simulation, locomotive DMI, braking advisory calculation, logging,
dashboard, .env configuration and documentation.
```
