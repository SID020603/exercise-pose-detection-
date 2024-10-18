import streamlit as st
import tensorflow as tf
import mediapipe as mp
import cv2

def arms():
    st.title("Arms")
    exercise = st.selectbox("Select an arm exercise", ["Bicep Curls", "Hammer Curls"])

    if st.button(f"Start {exercise}"):
        st.success('exercises started')
        model = tf.keras.models.load_model(r'C:\Users\siddh\OneDrive\Desktop\fitmentor_1\biceps\bicep_model.h5')

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        # Start video capture
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)  # Set width to 640 pixels
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

        def calculate_angle(p1, p2, p3):
            """
            Calculate the angle between three points using the law of cosines.
            p1, p2, p3 are tuples or lists with (x, y) coordinates.
            """
            a = np.array(p1)  # First point
            b = np.array(p2)  # Second point
            c = np.array(p3)  # Third point
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
                
            return angle



        def draw_landmarks(frame, landmarks):
            # Draw landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            for landmark in landmarks:
                # Get landmark coordinates
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw green circles

            # Optionally, draw connections between landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 

                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS, 
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )


        def count_curls(frame):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                draw_landmarks(frame,landmarks)

                # Extract relevant keypoints (e.g., shoulder, elbow, wrist)
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate the angle at the elbow
                elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                print('elbow_angle:{elbow_angle}')
                # Use your model to check if form is correct
                keypoints = []
                for landmark in landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                keypoints = np.array(keypoints[:36]).reshape(1, 1, 36)
                prediction = model.predict(keypoints)
                correct_form = prediction[0][0] > 0.5

                return elbow_angle, correct_form
            return None, False

        def generate_frames():
            reps = 0
            stage = None
            frame_count = 0

            while True:
                success, frame = camera.read()
                if not success:
                    break
                
                frame_count += 1

                if frame_count % 3 != 0:
                    continue 


                elbow_angle, correct_form = count_curls(frame)

                if elbow_angle is not None:
                    # Check if the angle is in the rep range (e.g., curling up and down)
                    if elbow_angle > 160:
                        stage = "down"
                    if elbow_angle < 40 and stage == 'down' and correct_form:
                        stage = "up"
                        reps += 1

                    # Display Correct or Incorrect based on the form
                    if correct_form:
                        cv2.putText(frame, 'Correct Form', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'Incorrect Form', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.putText(frame, f'Elbow Angle: {int(elbow_angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Overlay the rep count on the frame
                cv2.putText(frame, f'Reps: {reps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Stream the frame
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
