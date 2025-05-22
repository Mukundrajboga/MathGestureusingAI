import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import os
import time
from fpdf import FPDF
import unicodedata
import pyttsx3
import tempfile
from datetime import datetime
import json

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Create directories for saving answers and history
save_directory = "D:/Saved_Answers"
history_directory = "D:/Drawing_History"
os.makedirs(save_directory, exist_ok=True)
os.makedirs(history_directory, exist_ok=True)

# Streamlit Page Configuration
st.set_page_config(page_title="Math Gesture Solver", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .title {
            text-align: center;
            color: #2E4053;
            font-size: 30px;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            padding: 10px;
            font-size: 18px;
            font-weight: bold;
            background: linear-gradient(to right, #0073e6, #00c6ff);
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #005bb5, #0096d6);
        }
        .answer-box {
            border: 2px solid #0073e6;
            padding: 10px;
            border-radius: 10px;
            background-color: #f9f9f9;
            color: #333;
            font-size: 18px;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    theme = st.selectbox("Select Theme", ["Light", "Dark", "Blue"])
    drawing_color = st.color_picker("Drawing Color", "#000000")
    voice_feedback = st.checkbox("Enable Voice Feedback", value=True)
    auto_save = st.checkbox("Auto Save Drawings", value=True)
    export_format = st.multiselect("Export Format", ["PDF", "PNG", "JPG"], default=["PDF"])

# Streamlit UI Layout
col1, col2 = st.columns([4, 2])

with col1:
    st.markdown('<h1 class="title">📷 Math Gesture Solver</h1>', unsafe_allow_html=True)
    start_button = st.button('▶ Start Camera', key="start")
    stop_button = st.button('⏹ Stop Camera', key="stop")
    undo_button = st.button('↩ Undo', key="undo")
    redo_button = st.button('↪ Redo', key="redo")
    FRAME_WINDOW = st.image([], width=900)

with col2:
    st.markdown('<h1 class="title">📝 Answer</h1>', unsafe_allow_html=True)
    output_text_area = st.markdown('<div class="answer-box">AI Response Will Appear Here...</div>',
                                   unsafe_allow_html=True)
    similar_example_area = st.markdown('<div class="answer-box">Similar Example Will Appear Here...</div>',
                                       unsafe_allow_html=True)
    steps_area = st.markdown('<div class="answer-box">Step-by-Step Solution Will Appear Here...</div>',
                            unsafe_allow_html=True)

# Configure Gemini AI
genai.configure(api_key="AIzaSyDEqmNbBQoVPKUZmOvlJ1izTi9jB0Bz5VM")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam
cap = None

# Hand Detector Setup
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Create a whiteboard for drawing
canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255

# Drawing history
drawing_history = []
current_history_index = -1
prev_pos = None

# Convert hex color to BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

# Function to clean AI response text
def clean_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# Function to detect hands and fingers
def get_hand_info(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

# Drawing function with color support
def draw(info, prev_pos, canvas, img):
    fingers, lmList = info
    current_pos = None

    # Drawing Mode (Index Finger Up)
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        color = hex_to_bgr(drawing_color)
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), color, 10)
        cv2.circle(img, tuple(current_pos), 10, (0, 0, 255), -1)

    # Erase Mode (Thumb Up)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas[:] = 255

    # Save current state to history
    if auto_save and current_pos is not None:
        save_drawing_state(canvas.copy())

    return current_pos, canvas

# Save drawing state to history
def save_drawing_state(canvas_state):
    global current_history_index
    drawing_history.append(canvas_state.copy())
    current_history_index = len(drawing_history) - 1

# Undo last drawing action
def undo():
    global current_history_index, canvas
    if current_history_index > 0:
        current_history_index -= 1
        canvas = drawing_history[current_history_index].copy()
        return True
    return False

# Redo last undone action
def redo():
    global current_history_index, canvas
    if current_history_index < len(drawing_history) - 1:
        current_history_index += 1
        canvas = drawing_history[current_history_index].copy()
        return True
    return False

# Export drawing to multiple formats
def export_drawing(canvas, formats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for fmt in formats:
        if fmt == "PDF":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, "Math Gesture Solver - Drawing", ln=True, align="C")
            temp_img = f"{history_directory}/temp_{timestamp}.png"
            cv2.imwrite(temp_img, canvas)
            pdf.image(temp_img, x=10, y=20, w=190)
            pdf.output(f"{save_directory}/drawing_{timestamp}.pdf")
            os.remove(temp_img)
        elif fmt == "PNG":
            cv2.imwrite(f"{save_directory}/drawing_{timestamp}.png", canvas)
        elif fmt == "JPG":
            cv2.imwrite(f"{save_directory}/drawing_{timestamp}.jpg", canvas)

# AI Function to solve drawn equations with step-by-step solution
def send_to_ai(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # Four fingers up
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem step by step", pil_image])
        answer_text = response.text

        # Extract steps from the response
        steps = answer_text.split('\n')
        steps_text = '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(steps) if step.strip()])

        # Ask AI to provide a similar example
        similar_response = model.generate_content(f"Give a similar example to this problem and solve it: {answer_text}")
        similar_text = similar_response.text if similar_response else "No example available."

        # Clean text for PDF
        xyz = clean_text(answer_text)
        xyz1 = clean_text(similar_text)
        steps_clean = clean_text(steps_text)

        # Save with UTF-8 encoding
        filename = f"answer_{int(time.time())}.pdf"
        file_path = os.path.join(save_directory, filename)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", style='', size=12)

        pdf.cell(200, 10, "Math Gesture Solver - AI Response", ln=True, align="C")
        pdf.ln(10)
        
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Solution:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, xyz)

        pdf.ln(5)
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Step-by-Step Solution:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, steps_clean)

        pdf.ln(5)
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Similar Example:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, xyz1)

        pdf.output(file_path)

        # Export drawing in selected formats
        export_drawing(canvas, export_format)

        # Update UI
        st.success(f"✅ Answer saved at: {file_path}")
        output_text_area.empty()
        output_text_area.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)

        similar_example_area.empty()
        similar_example_area.markdown(f'<div class="answer-box">{similar_text}</div>', unsafe_allow_html=True)

        steps_area.empty()
        steps_area.markdown(f'<div class="answer-box">{steps_text}</div>', unsafe_allow_html=True)

        # Voice feedback if enabled
        if voice_feedback:
            engine.say("Here's the solution to your math problem")
            engine.runAndWait()
            engine.say(answer_text)
            engine.runAndWait()

        return answer_text, similar_text, steps_text

    return "", "", ""

# Streamlit Application Loop
if start_button:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while cap.isOpened():
        success, img = cap.read()
        if not success or img is None:
            st.error("❌ Failed to capture image from webcam. Check camera connection.")
            break

        img = cv2.flip(img, 1)

        info = get_hand_info(img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas, img)
            send_to_ai(model, canvas, fingers)

        # Handle undo/redo
        if undo_button:
            if undo():
                st.success("↩ Undo successful")
            else:
                st.warning("No more actions to undo")
        if redo_button:
            if redo():
                st.success("↪ Redo successful")
            else:
                st.warning("No more actions to redo")

        # Overlay the whiteboard onto the webcam feed
        blended_img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

        # Show output in Streamlit
        FRAME_WINDOW.image(blended_img, channels="BGR")

        # Stop the loop if stop button is pressed
        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()