import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import cvzone
import requests
import smtplib
import imghdr
from email.message import EmailMessage
import pandas as pd
import time
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Load YOLO models
violenceDetect_model = YOLO("VoilenceDetection/best.pt")
person_model = YOLO("yolov8n.pt")

classNames = ['NonViolence', 'Violence']

# Thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5
VIOLENCE_CONFIDENCE_THRESHOLD = 0.85

# Email Configuration
EMAIL_SENDER = "21071a6674@vnrvjiet.in"
EMAIL_PASSWORD = "qvqv knzi wcun ehkd"

# WhatsApp API
INSTANCE_ID = "instance118173"
API_TOKEN = "dx52fdq2cvyczxmy"
API_URL = f"https://api.ultramsg.com/{INSTANCE_ID}/messages/chat"

# Load or create personnel data
CSV_FILE = "personnel_data.csv"
try:
    personnel_df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    personnel_df = pd.DataFrame(columns=["Name", "Phone", "Email"])
    personnel_df.to_csv(CSV_FILE, index=False)

# Email alert
def send_email_alert(subject, body, recipient_email, image_path):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = recipient_email
    msg.set_content(body)

    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_type = imghdr.what(img_file.name)
        msg.add_attachment(img_data, maintype="image", subtype=img_type, filename="alert.jpg")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

# WhatsApp alert

def send_whatsapp_alert(phone, message):
    phone = str(phone).strip().split('.')[0]  # Convert float to string and remove decimal if any

    if not phone.startswith("+"):
        phone = "+" + phone  # Replace +91 with your actual country code

    payload = {
        "token": API_TOKEN,
        "to": f"whatsapp:{phone}",
        "body": message
    }

    print(payload)
    
    try:
        response = requests.post(API_URL, data=payload)
        if response.status_code != 200:
            st.error(f"❌ WhatsApp alert failed: {response.status_code} - {response.text}")
        else:
            st.success(f"✅ WhatsApp alert sent to {phone}")
    except Exception as e:
        st.error(f"🚨 Exception while sending WhatsApp alert: {e}")


# Alert all
def send_alerts(frame):
    frame_path = "violence_alert.jpg"
    cv2.imwrite(frame_path, frame)
    for _, row in personnel_df.iterrows():
        #send_email_alert("Violence Detected", "Alert! Violence detected in the premises.", row["Email"], frame_path)
        send_whatsapp_alert(row["Phone"], "Alert! Violence detected in the premises. Check your email for details.")

# UI - Personnel registration
st.sidebar.title("Personnel Registration")
with st.sidebar.form("register_form"):
    name = st.text_input("Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email Address")
    submit = st.form_submit_button("Register")

    if submit and name and phone and email:
        new_entry = pd.DataFrame([[name, phone, email]], columns=["Name", "Phone", "Email"])
        personnel_df = pd.concat([personnel_df, new_entry], ignore_index=True)
        personnel_df.to_csv(CSV_FILE, index=False)
        st.success("Registration Successful!")

# App
st.title("Violence Detection System")
option = st.radio("Choose Input Source:", ("Upload Video", "Live Webcam"))

cap = None

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

elif option == "Live Webcam":
    # Try multiple indices in case one fails
    for i in range(3):
        temp_cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(3)  # Longer warm-up time
        if temp_cap.isOpened():
            cap = temp_cap
            st.success(f"✅ Camera index {i} opened successfully.")
            break
        else:
            temp_cap.release()
    if not cap or not cap.isOpened():
        st.error("🚫 Could not open camera. It may be in use by another application.")

if cap:
    stframe = st.empty()
    fail_count = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success or frame is None:
            fail_count += 1
            st.warning("⚠️ Could not read from webcam. Showing black frame.")
            black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            stframe.image(black_frame, channels="RGB")
            if fail_count > 30:
                st.error("📷 Webcam feed failed repeatedly. Please check camera permissions or restart.")
                break
            continue
        else:
            fail_count = 0

        person_results = person_model(frame)
        persons = []
        for result in person_results:
            for box in result.boxes:
                conf = box.conf[0]
                if conf > PERSON_CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append((x1, y1, x2, y2))

        if persons:
            violence_results = violenceDetect_model(frame)
            for result in violence_results:
                for box in result.boxes:
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    if conf > VIOLENCE_CONFIDENCE_THRESHOLD and currentClass == 'Violence':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        for px1, py1, px2, py2 in persons:
                            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cvzone.putTextRect(frame, f'Violence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                                send_alerts(frame)
                                break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
