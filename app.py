# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import time
from pathlib import Path
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


class SignLanguageProcessor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.actions = ['hello', 'please', 'thumbs up']
        self.sequence_length = 20
        self.sequence = []
        self.current_action = ""
        self.threshold = 0.4

        # Load model
        model_path = Path(__file__).parent / "sign_language_model3.h5"
        self.model = load_model(model_path)

        # Initialize holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, results):
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        left_hand = np.array([[res.x, res.y, res.z] for res in
                              results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            63)
        right_hand = np.array([[res.x, res.y, res.z] for res in
                               results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            63)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        return np.concatenate([face, left_hand, right_hand, pose])

    def draw_styled_landmarks(self, image, results):
        # Draw face landmarks
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, None,
            self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )

        # Draw hand landmarks
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        self.draw_styled_landmarks(image, results)

        # Make prediction
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.sequence_length:]

        if len(self.sequence) == self.sequence_length:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            predicted_idx = np.argmax(res)

            if res[predicted_idx] > self.threshold:
                self.current_action = self.actions[predicted_idx]

                # Draw prediction on frame
                cv2.putText(image, f"Detected: {self.current_action}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Confidence: {res[predicted_idx]:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

        return image


class SignLanguageStreamer:
    def __init__(self):
        self.processor = SignLanguageProcessor()

    def callback(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.processor.process_frame(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    # Set page config
    st.set_page_config(page_title="Sign Language Recognition",
                       page_icon="👋",
                       layout="wide")

    # Add title and description
    st.title("Real-time Sign Language Recognition")
    st.markdown("""
    This application recognizes the following signs in real-time:
    - Hello
    - Please
    - Thumbs Up

    Make sure to allow camera access when prompted.
    """)

    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Create webRTC streamer
    streamer = SignLanguageStreamer()
    webrtc_streamer(
        key="sign-language",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=streamer.callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


if __name__ == "__main__":
    main()