"""Streamlit client for the PathSnap API."""

import os

import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")
st.title("PathSnap")
st.caption("Explore GPS-to-road map matching predictions.")

with st.sidebar:
    st.header("API connection")
    api_url = st.text_input(
        "API base URL",
        value=os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"),
        help="For two laptops, use the API host's LAN IP, for example http://192.168.1.20:8000.",
    ).rstrip("/")
    timeout_s = st.number_input("Request timeout (seconds)", min_value=30, max_value=1800,
                                value=600, step=30)

try:
    response = requests.get(f"{api_url}/trips", timeout=15)
    response.raise_for_status()
    trip_ids = response.json().get("trips", [])
except requests.RequestException as exc:
    st.error(f"Could not reach the API at {api_url}: {exc}")
    st.info("Start the API with `uvicorn api.app:app --host 0.0.0.0 --port 8000`.")
    st.stop()

if not trip_ids:
    st.warning("The API returned no trips. Check the trip_root in configs/api.yaml.")
    st.stop()

trip_id = st.selectbox("Student test trip", trip_ids)
if st.button("Predict route", type="primary"):
    with st.spinner(f"Predicting trip {trip_id}…"):
        try:
            prediction = requests.post(f"{api_url}/predict", json={"trip_id": trip_id},
                                       timeout=float(timeout_s))
            prediction.raise_for_status()
            result = prediction.json()
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", "")
            st.error(f"Prediction request failed: {exc}\n{detail}")
            st.stop()

    st.subheader(f"Trip {trip_id}")
    metrics = result.get("metrics", {})
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [
        ("HMM top-1 accuracy", "top1_accuracy"),
        ("Trusted top-1 accuracy", "top1_accuracy_trusted"),
        ("Route continuity", "route_continuity"),
        ("Candidate miss rate", "candidate_miss_rate"),
    ]):
        value = metrics.get(key)
        col.metric(label, "—" if value is None else f"{value:.1%}")

    st.caption(f"{metrics.get('observations', 0):,} observations · "
               f"{len(result.get('predicted_arcs', [])):,} decoded arcs · "
               f"{result.get('timing', {}).get('seconds', 0):.2f}s")
    map_html = result.get("route_geojson")
    if map_html:
        components.html(map_html, height=680, scrolling=True)
    else:
        st.warning("The API response did not include a route map.")
