#!/usr/bin/env python3
"""
Streamlit app for bike share demand prediction visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from streamlit_folium import st_folium
import folium
import h3

# Page configuration
st.set_page_config(page_title="Bike Share Demand Predictions", layout="wide")

# File paths
PREDICTIONS_CSV = Path("result/chicago_h3_predictions_res8.csv")
H3_RESOLUTION = 8

# Load predictions data
@st.cache_data
def load_predictions():
    """Load predictions CSV file."""
    if not PREDICTIONS_CSV.exists():
        st.error(f"Predictions file not found: {PREDICTIONS_CSV}")
        st.stop()
    return pd.read_csv(PREDICTIONS_CSV)

# Load data
predictions_df = load_predictions()

# Title
st.title("Bike Share Demand Predictions")
st.markdown("Click on the map to see predictions for that location.")

# Initialize session state
if 'lat' not in st.session_state:
    st.session_state.lat = 41.8781
    st.session_state.lng = -87.6298
    st.session_state.zoom = 12
    st.session_state.center = [41.8781, -87.6298]

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    # Create map with preserved zoom and center
    m = folium.Map(
        location=st.session_state.center,
        zoom_start=st.session_state.zoom
    )
    
    # Add marker for selected location with visible icon
    folium.Marker(
        [st.session_state.lat, st.session_state.lng],
        popup=f"Selected Location<br>Lat: {st.session_state.lat:.6f}, Lng: {st.session_state.lng:.6f}",
        tooltip="Selected Location",
        icon=folium.Icon(color='red', icon='map-marker', prefix='fa')
    ).add_to(m)
    
    # Also add a circle marker for better visibility
    folium.CircleMarker(
        location=[st.session_state.lat, st.session_state.lng],
        radius=10,
        popup=f"Selected Location<br>Lat: {st.session_state.lat:.6f}, Lng: {st.session_state.lng:.6f}",
        color='red',
        fill=True,
        fillColor='red',
        fillOpacity=0.6
    ).add_to(m)
    
    # Display map and get zoom/center info
    map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked", "zoom", "center"])
    
    # Preserve zoom and center from map
    if "zoom" in map_data and map_data["zoom"] is not None:
        st.session_state.zoom = map_data["zoom"]
    if "center" in map_data and map_data["center"] is not None:
        st.session_state.center = [map_data["center"]["lat"], map_data["center"]["lng"]]
    
    # Handle map click - only update lat/lng, keep zoom and center
    # Only respond to map clicks (not marker clicks)
    if map_data.get("last_clicked") is not None:
        clicked = map_data["last_clicked"]
        st.session_state.lat = clicked["lat"]
        st.session_state.lng = clicked["lng"]
        st.session_state.selected_station = None
        st.rerun()

# Get H3 index
try:
    h3_index = h3.latlng_to_cell(st.session_state.lat, st.session_state.lng, H3_RESOLUTION)
except Exception as e:
    st.error(f"Error: {e}")
    h3_index = None

with col2:
    # Get predictions
    if h3_index:
        h3_predictions = predictions_df[predictions_df['h3_index'] == h3_index].copy()
        
        if len(h3_predictions) > 0:
            h3_predictions = h3_predictions.sort_values('month')
            
            # Create chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=h3_predictions['month'],
                y=h3_predictions['cbike_start'],
                mode='lines+markers',
                name='Classic Bike Start'
            ))
            
            fig.add_trace(go.Scatter(
                x=h3_predictions['month'],
                y=h3_predictions['cbike_end'],
                mode='lines+markers',
                name='Classic Bike End'
            ))
            
            fig.add_trace(go.Scatter(
                x=h3_predictions['month'],
                y=h3_predictions['ebike_start'],
                mode='lines+markers',
                name='E-Bike Start'
            ))
            
            fig.add_trace(go.Scatter(
                x=h3_predictions['month'],
                y=h3_predictions['ebike_end'],
                mode='lines+markers',
                name='E-Bike End'
            ))
            
            fig.update_layout(
                title="Average predicted daily trips for each month",
                xaxis_title="Month",
                yaxis_title="Predicted Trips",
                height=500
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No predictions found for this location.")
    else:
        st.info("Select a location on the map.")
