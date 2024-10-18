import streamlit as st
# from utils.pose_detection import start_camera_for_exercise

def legs():
    st.title("Legs")
    exercise = st.selectbox("Select a leg exercise", ["Squats", "Lunges"])

    if st.button(f"Start {exercise}"):
        st.success('exercise starts')
