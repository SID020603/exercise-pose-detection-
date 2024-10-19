import streamlit as st
import cv2
import tempfile
import tensorflow as tf
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os

# Load the trained model
@st.cache_resource
def load_model(exercise):
    model_path = os.path.join(os.path.dirname(__file__), f'{exercise.lower()}_model.h5')
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info(f"Please make sure the '{exercise.lower()}_model.h5' file is in the same directory as this script.")
        return None
    return tf.keras.models.load_model(model_path)

class ExerciseProcessor(VideoProcessorBase):
    def __init__(self, exercise_type, exercise):
        self.exercise_type = exercise_type
        self.exercise = exercise
        self.model = load_model(exercise)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.reps = 0
        self.stage = None
        self.correct_form_count = 0
        self.total_frames = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame and get exercise metrics
        img, metrics = self.process_frame(img)
        
        # Display metrics on the frame
        self.display_metrics(img, metrics)
        
        return img

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Extract keypoints for the model
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            keypoints = np.array(keypoints).reshape(1, 1, -1)

            # Predict form correctness
            prediction = self.model.predict(keypoints)[0]
            correct_form = np.argmax(prediction) == 1  # Assuming 1 is the index for correct form

            # Count reps and track form
            if self.exercise == "Squats":
                self.count_squats(results.pose_landmarks.landmark, correct_form)
            elif self.exercise == "Lunges":
                self.count_lunges(results.pose_landmarks.landmark, correct_form)

            self.total_frames += 1
            if correct_form:
                self.correct_form_count += 1

            form_accuracy = (self.correct_form_count / self.total_frames) * 100 if self.total_frames > 0 else 0

            knee_angle = self.calculate_knee_angle(results.pose_landmarks.landmark)

            return frame, {
                "reps": self.reps,
                "form_accuracy": form_accuracy,
                "correct_form": correct_form,
                "knee_angle": knee_angle
            }
        
        return frame, {}

    def calculate_knee_angle(self, landmarks):
        def get_coordinates(landmark):
            return [landmark.x, landmark.y]

        hip = get_coordinates(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])
        knee = get_coordinates(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])
        ankle = get_coordinates(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])

        radians = np.arctan2(ankle[1] - knee[1], ankle[0] - knee[0]) - \
                  np.arctan2(hip[1] - knee[1], hip[0] - knee[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
                
        return angle

    def count_squats(self, landmarks, correct_form):
        knee_angle = self.calculate_knee_angle(landmarks)

        if knee_angle > 160:
            self.stage = "up"
        if knee_angle < 90 and self.stage == 'up' and correct_form:
            self.stage = "down"
            self.reps += 1

    def count_lunges(self, landmarks, correct_form):
        knee_angle = self.calculate_knee_angle(landmarks)

        if knee_angle > 160:
            self.stage = "up"
        if knee_angle < 90 and self.stage == 'up' and correct_form:
            self.stage = "down"
            self.reps += 1

    def display_metrics(self, frame, metrics):
        if metrics:
            cv2.putText(frame, f"Reps: {metrics['reps']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Form Accuracy: {metrics['form_accuracy']:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            form_text = "Correct Form" if metrics['correct_form'] else "Incorrect Form"
            form_color = (0, 255, 0) if metrics['correct_form'] else (0, 0, 255)
            cv2.putText(frame, form_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, form_color, 2)
            cv2.putText(frame, f"Knee Angle: {int(metrics['knee_angle'])}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def process_uploaded_video(video_file, exercise):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    processor = ExerciseProcessor("Legs", exercise)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, _ = processor.process_frame(frame)
        frames.append(processed_frame)
    
    cap.release()
    os.unlink(tfile.name)
    
    return frames, processor.reps, processor.correct_form_count / processor.total_frames * 100 if processor.total_frames > 0 else 0

def legs(video_option, uploaded_file=None):
    st.title("Leg Exercises")
    exercise = st.selectbox("Select a leg exercise", ["Squats", "Lunges"])

    st.write(f"Let's do some {exercise}!")
    st.write("Position yourself so that your full body is visible in the camera.")

    model = load_model(exercise)
    if model is None:
        st.error("Cannot proceed without the model. Please resolve the issue and try again.")
        return
    
    if video_option == "Live Webcam":
        webrtc_ctx = webrtc_streamer(
            key="leg-exercise",
            video_processor_factory=lambda: ExerciseProcessor("Legs", exercise),
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
        )

        if webrtc_ctx.video_processor:
            st.write(f"Start your {exercise} now!")
            st.write("The AI will count your reps and provide feedback on your form.")
    
    elif video_option == "Upload Video":
        if uploaded_file is not None:
            st.video(uploaded_file)
            if st.button("Process Video"):
                with st.spinner('Processing video...'):
                    frames, reps, form_accuracy = process_uploaded_video(uploaded_file, exercise)
                st.success('Video processed!')
                st.write(f"Reps counted: {reps}")
                st.write(f"Form accuracy: {form_accuracy:.2f}%")
                
                # Display processed frames as a video
                st.write("Processed Video:")
                stframe = st.empty()
                for frame in frames:
                    stframe.image(frame, channels="BGR")
        else:
            st.warning("Please upload a video file.")
    
    # Add some tips for the exercise
    st.markdown("---")
    st.subheader("Tips for perfect form:")
    if exercise == "Squats":
        st.write("- Keep your feet shoulder-width apart")
        st.write("- Lower your body as if sitting back into a chair")
        st.write("- Keep your chest up and your weight on your heels")
    elif exercise == "Lunges":
        st.write("- Step forward with one leg, lowering your hips")
        st.write("- Keep your front knee directly above your ankle")
        st.write("- Push back up to the starting position and repeat with the other leg")

# Run the main function
if __name__ == "__main__":
    legs()
