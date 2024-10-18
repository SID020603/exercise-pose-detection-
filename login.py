import streamlit as st

def login_page():
    st.title("Fitness Tracker Login")

    option = st.selectbox("Login or Signup", ["","Login", "Signup", "Login as Guest"])

    if option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username and password:  # Simple check
                st.success("Logged in successfully")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Please enter a username and password")

    elif option == "Signup":
        username = st.text_input("Create a username")
        password = st.text_input("Create a password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        if st.button("Signup"):
            if password == confirm_password:
                st.success("Signed up successfully")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Passwords do not match")

    elif option == "Login as Guest":
        st.session_state["logged_in"] = True
        st.session_state["username"] = "Guest"
        st.success("Logged in as Guest")
        st.rerun()

# To import this function in the main app later.
