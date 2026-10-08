import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from utils.structural import FEATURES, predict_structural, train_model
from utils.fusion import structural_risk, lidar_risk, vision_risk, fused_risk, risk_level
from utils.braking import intervention_decision
from utils.lora import make_packet, relay_packet
from utils.vision import detect_intrusions
from utils.events import log_event, read_events

load_dotenv()

APP_NAME = os.getenv("APP_NAME","RailKavach Edge AI")
CORRIDOR_NAME = os.getenv("CORRIDOR_NAME","Demo Mountain Corridor")
LIDAR_THRESHOLD = float(os.getenv("LIDAR_CLEARANCE_THRESHOLD_M","1.8"))
BRAKE_DECEL = float(os.getenv("BRAKE_DECELERATION_MPS2","0.65"))
REACTION_BUFFER = float(os.getenv("REACTION_BUFFER_S","5"))
DEFAULT_SPEED = float(os.getenv("DEFAULT_TRAIN_SPEED_KMPH","70"))

BASE = Path(__file__).resolve().parent
STRUCTURAL_DATA = BASE/"data"/"structural_sensor_data.csv"
NODES_DATA = BASE/"data"/"trackside_nodes.csv"

st.set_page_config(page_title=APP_NAME,page_icon="🚆",layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(STRUCTURAL_DATA), pd.read_csv(NODES_DATA)

@st.cache_resource
def load_model():
    return train_model(False)

sensor_df, nodes_df = load_data()
model, metrics = load_model()

with st.sidebar:
    st.title(APP_NAME)
    st.caption(f"Offline V2I rail safety prototype • {CORRIDOR_NAME}")
    page = st.radio("Navigation",[
        "Command Dashboard",
        "Trackside Edge Node",
        "Structural Health Monitor",
        "LiDAR Clearance Monitor",
        "Vision Intrusion Detection",
        "LoRa V2I Mesh",
        "Locomotive DMI Simulator",
        "Event Analytics",
        "Model Performance"
    ])
    st.divider()
    st.warning("Prototype only — not for real railway signalling, braking, or safety-critical deployment.")

if page == "Command Dashboard":
    st.title("RailKavach Edge AI Command Dashboard")
    st.subheader("Decentralized offline hazard interception for vulnerable rail corridors")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Trackside Nodes",len(nodes_df))
    c2.metric("Node Spacing",f"{os.getenv('NODE_SPACING_M','500')} m")
    c3.metric("Structural ML Accuracy",f"{metrics['accuracy']*100:.1f}%")
    c4.metric("Network Mode","Offline Mesh")

    st.subheader("Protected Corridor Nodes")
    st.map(nodes_df.rename(columns={"lat":"latitude","lon":"longitude"})[["latitude","longitude"]])

    st.dataframe(nodes_df,use_container_width=True,hide_index=True)

    st.subheader("Architecture")
    st.write(
        "Each edge node fuses structural vibration sensing, geometric clearance ranging, "
        "and optional local computer vision. When risk crosses a threshold, the software "
        "simulates a local acoustic deterrent and transmits a compact hazard packet through "
        "an offline Sub-GHz mesh to the locomotive-side receiver."
    )

elif page == "Trackside Edge Node":
    st.title("Trackside Edge Node — Multi-Modal Sensor Fusion")

    node_id = st.selectbox("Trackside Node",nodes_df["node_id"].tolist())
    node = nodes_df[nodes_df["node_id"]==node_id].iloc[0]

    st.caption(f"{node['zone']} • {node['terrain']} • Chainage {node['chainage_km']} km")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.subheader("Structural Sensors")
        rms = st.number_input("RMS Vibration (mm/s)",0.1,20.0,2.0,0.1)
        peak = st.number_input("Peak Acceleration (g)",0.1,20.0,3.0,0.1)
        f1 = st.number_input("Dominant Frequency 1 (Hz)",1.0,400.0,85.0,1.0)
        f2 = st.number_input("Dominant Frequency 2 (Hz)",1.0,500.0,165.0,1.0)
        temp = st.number_input("Rail Temperature (°C)",-20.0,90.0,35.0,1.0)
        gauge = st.number_input("Gauge Deviation (mm)",0.0,20.0,0.8,0.1)
        ballast = st.number_input("Ballast Settlement (mm)",0.0,40.0,2.5,0.5)

    with c2:
        st.subheader("LiDAR / Clearance")
        clearance = st.number_input("Nearest Track Intrusion Distance (m)",0.1,20.0,5.0,0.1)

        st.subheader("Vision Sensor")
        vision_object = st.selectbox(
            "Simulated Vision Detection",
            ["None","Person","Car","Truck","Bus","Motorcycle","Bicycle","Cow","Horse","Dog","Elephant","Debris"]
        )

    with c3:
        st.subheader("Train Context")
        train_distance = st.number_input("Train Distance to Node (m)",50.0,10000.0,2500.0,50.0)
        speed = st.number_input("Train Speed (km/h)",0.0,160.0,DEFAULT_SPEED,5.0)
        relay_hops = st.slider("Mesh Relay Hops",1,8,3)

    if st.button("Run Edge Fusion & V2I Response",use_container_width=True):
        row = pd.DataFrame([{
            "rms_vibration_mm_s":rms,
            "peak_acceleration_g":peak,
            "dominant_freq_1_hz":f1,
            "dominant_freq_2_hz":f2,
            "rail_temperature_c":temp,
            "gauge_deviation_mm":gauge,
            "ballast_settlement_mm":ballast
        }])
        structural = predict_structural(row).iloc[0]
        s_score = structural_risk(structural)
        l_score = lidar_risk(clearance,LIDAR_THRESHOLD)

        sim_detections = []
        if vision_object != "None":
            sim_detections = [{"label":vision_object.lower(),"confidence":0.96}]
        v_score = 80 if vision_object == "Debris" else vision_risk(sim_detections)

        total = fused_risk(s_score,l_score,v_score)
        level = risk_level(total)

        fault = structural["predicted_fault"]
        hazards = []
        if fault != "Healthy":
            hazards.append(fault)
        if l_score >= 70:
            hazards.append("Physical Clearance Breach")
        if vision_object != "None":
            hazards.append(f"Vision: {vision_object}")
        hazard_type = " + ".join(hazards) if hazards else "No verified hazard"

        a,b,c,d = st.columns(4)
        a.metric("Structural Risk",f"{s_score}/100")
        b.metric("LiDAR Risk",f"{l_score}/100")
        c.metric("Vision Risk",f"{v_score}/100")
        d.metric("Fused Risk",f"{total}/100",level)

        st.write(f"Structural classification: {fault} ({structural['confidence_pct']:.1f}% confidence)")

        decision = intervention_decision(
            train_distance,speed,level,BRAKE_DECEL,REACTION_BUFFER
        )

        packet = make_packet(
            node_id,node["chainage_km"],total,level,hazard_type,train_distance
        )
        mesh = relay_packet(packet,relay_hops)

        if level in ["HIGH","CRITICAL"]:
            st.error(f"LOCAL RESPONSE: Acoustic deterrent simulation ACTIVE • {decision['action']}")
        elif level == "MODERATE":
            st.warning(f"LOCAL RESPONSE: Heightened monitoring • {decision['action']}")
        else:
            st.success("No intervention required. Continue monitoring.")

        st.subheader("Locomotive Intervention Calculation")
        x,y,z = st.columns(3)
        x.metric("Estimated Stopping Distance",f"{decision['stopping_distance_m']} m")
        y.metric("Time to Hazard",f"{decision['time_to_hazard_s']} s")
        z.metric("Safety Margin",f"{decision['margin_m']} m")
        st.code(mesh["packet"],language="json")

        log_event({
            "node_id":node_id,
            "zone":node["zone"],
            "hazard_type":hazard_type,
            "risk_score":total,
            "risk_level":level,
            "train_distance_m":train_distance,
            "train_speed_kmph":speed,
            "recommended_action":decision["action"]
        })

elif page == "Structural Health Monitor":
    st.title("Structural Health Monitoring")

    with st.form("structural_form"):
        c1,c2 = st.columns(2)
        with c1:
            rms = st.number_input("RMS Vibration (mm/s)",0.1,20.0,2.0,0.1,key="s_rms")
            peak = st.number_input("Peak Acceleration (g)",0.1,20.0,3.0,0.1,key="s_peak")
            f1 = st.number_input("Dominant Frequency 1 (Hz)",1.0,400.0,85.0,1.0,key="s_f1")
            f2 = st.number_input("Dominant Frequency 2 (Hz)",1.0,500.0,165.0,1.0,key="s_f2")
        with c2:
            temp = st.number_input("Rail Temperature (°C)",-20.0,90.0,35.0,1.0,key="s_temp")
            gauge = st.number_input("Gauge Deviation (mm)",0.0,20.0,0.8,0.1,key="s_gauge")
            ballast = st.number_input("Ballast Settlement (mm)",0.0,40.0,2.5,0.5,key="s_ballast")
        go = st.form_submit_button("Analyze Structural Health",use_container_width=True)

    if go:
        row = pd.DataFrame([{
            "rms_vibration_mm_s":rms,
            "peak_acceleration_g":peak,
            "dominant_freq_1_hz":f1,
            "dominant_freq_2_hz":f2,
            "rail_temperature_c":temp,
            "gauge_deviation_mm":gauge,
            "ballast_settlement_mm":ballast
        }])
        result = predict_structural(row).iloc[0]
        score = structural_risk(result)
        st.metric("Predicted Track Condition",result["predicted_fault"])
        st.metric("Model Confidence",f"{result['confidence_pct']:.1f}%")
        st.metric("Structural Risk",f"{score}/100")

        if result["predicted_fault"] == "Rail Fracture":
            st.error("Possible rail-fracture pattern detected in prototype data.")
        elif result["predicted_fault"] != "Healthy":
            st.warning("Abnormal structural pattern detected.")
        else:
            st.success("Sensor pattern classified as healthy.")

elif page == "LiDAR Clearance Monitor":
    st.title("LiDAR Clearance Monitoring")
    st.caption("Prototype geometric intrusion logic using simulated minimum clearance.")

    distance = st.slider("Nearest object distance from protected rail envelope (m)",0.1,10.0,3.5,0.1)
    score = lidar_risk(distance,LIDAR_THRESHOLD)
    level = risk_level(score)

    c1,c2,c3 = st.columns(3)
    c1.metric("Measured Clearance",f"{distance:.1f} m")
    c2.metric("Configured Threshold",f"{LIDAR_THRESHOLD:.1f} m")
    c3.metric("LiDAR Risk",f"{score}/100",level)

    if score >= 70:
        st.error("Potential physical breach of the protected clearance envelope.")
    elif score >= 35:
        st.warning("Object is approaching the configured clearance envelope.")
    else:
        st.success("Clearance currently within prototype safe range.")

elif page == "Vision Intrusion Detection":
    st.title("Edge Vision Intrusion Detection")
    st.caption("Optional YOLO-based image analysis. First use may download the model if it is not installed.")

    upload = st.file_uploader("Upload optical/thermal-style track image",type=["jpg","jpeg","png"])
    if upload:
        image = Image.open(upload).convert("RGB")
        frame = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

        try:
            annotated,detections = detect_intrusions(frame)
            st.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB),use_container_width=True)
            score = vision_risk(detections)
            st.metric("Vision Risk",f"{score}/100",risk_level(score))
            if detections:
                st.dataframe(pd.DataFrame(detections),use_container_width=True,hide_index=True)
            else:
                st.success("No configured intrusion class detected.")
        except Exception as e:
            st.error(f"Vision model could not run: {e}")
            st.info("The rest of the project works without the vision model.")

elif page == "LoRa V2I Mesh":
    st.title("Offline Sub-GHz V2I Mesh Simulation")
    st.write(
        "This page simulates a compact hazard packet being relayed between fixed trackside nodes "
        "without cellular connectivity. It does not transmit real radio packets."
    )

    node_id = st.selectbox("Source Node",nodes_df["node_id"].tolist(),key="lora_node")
    risk_score = st.slider("Risk Score",0,100,85)
    hazard = st.text_input("Hazard Type",value="Track intrusion")
    distance = st.number_input("Distance to Train (m)",100.0,20000.0,3000.0,100.0)
    hops = st.slider("Relay Hops",1,10,4,key="lora_hops")
    node = nodes_df[nodes_df["node_id"]==node_id].iloc[0]

    if st.button("Transmit Hazard Packet",use_container_width=True):
        level = risk_level(risk_score)
        packet = make_packet(node_id,node["chainage_km"],risk_score,level,hazard,distance)
        mesh = relay_packet(packet,hops)
        st.success("Packet delivered in simulation.")
        st.json(mesh)

elif page == "Locomotive DMI Simulator":
    st.title("Locomotive DMI Hazard Countdown Simulator")
    st.caption("Software-only demonstration of advisory/braking decision logic. No train-control interface is included.")

    distance = st.number_input("Distance to Hazard (m)",50.0,20000.0,4000.0,50.0)
    speed = st.number_input("Train Speed (km/h)",0.0,160.0,80.0,5.0)
    risk = st.selectbox("Incoming Hazard Level",["LOW","MODERATE","HIGH","CRITICAL"])

    decision = intervention_decision(distance,speed,risk,BRAKE_DECEL,REACTION_BUFFER)

    a,b,c = st.columns(3)
    a.metric("Distance to Hazard",f"{distance:.0f} m")
    b.metric("Time to Hazard",f"{decision['time_to_hazard_s']} s")
    c.metric("Estimated Stop Distance",f"{decision['stopping_distance_m']} m")

    if decision["action"] == "EMERGENCY BRAKE COMMAND":
        st.error(decision["action"])
    elif "BRAK" in decision["action"] or "SPEED" in decision["action"]:
        st.warning(decision["action"])
    else:
        st.info(decision["action"])

    st.metric("Distance Margin",f"{decision['margin_m']} m")

elif page == "Event Analytics":
    st.title("Hazard Event Analytics")
    events = read_events()
    if events.empty:
        st.info("No events logged yet. Run the Trackside Edge Node simulation first.")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Events",len(events))
        c2.metric("Critical Events",int((events["risk_level"]=="CRITICAL").sum()))
        c3.metric("High Events",int((events["risk_level"]=="HIGH").sum()))

        by_level = events.groupby("risk_level",as_index=False).size()
        st.plotly_chart(px.bar(by_level,x="risk_level",y="size",text_auto=True),use_container_width=True)
        st.dataframe(events.sort_values("timestamp",ascending=False),use_container_width=True,hide_index=True)

elif page == "Model Performance":
    st.title("Structural ML Model Performance")
    c1,c2 = st.columns(2)
    c1.metric("Accuracy",f"{metrics['accuracy']*100:.1f}%")
    c2.metric("Macro F1",f"{metrics['macro_f1']:.3f}")
    st.write(f"Training rows: {metrics['train_rows']}")
    st.write(f"Test rows: {metrics['test_rows']}")
    st.warning(
        "These figures come from synthetic demonstration data. They are not validated railway-safety performance metrics."
    )
