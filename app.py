import streamlit as st
from login import login_page
from streamlit_option_menu import option_menu
from arms import arms
from legs import legs
import cv2

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def main():
    # If not logged in, show login/signup page
    if not st.session_state["logged_in"]:
        login_page()
    else:
        # Once logged in, show the muscle group selection
        st.sidebar.title(f"Welcome, {st.session_state['username']}")
        st
        st.title("Select Muscle Group")
        muscle_group = option_menu("Select a Muscle Group", 
                                   ["Chest", "Arms", "Legs", "Back", "Shoulders"], 
                                   icons=["heart", "handbag", "shoe", "back", "tree"],
                                   default_index=0)
        if muscle_group == "Arms":
            arms()
        elif muscle_group == "Legs":
            legs()
        # Add similar for other muscle groups

# Run the main function
if __name__ == "__main__":
    main()
