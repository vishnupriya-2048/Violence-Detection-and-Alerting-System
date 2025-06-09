import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import cvzone
import smtplib
import imghdr
from email.message import EmailMessage

# Load YOLO models
violenceDetect_model = YOLO("VoilenceDetection/best.pt")
person_model = YOLO("yolov8n.pt")

classNames = ['NonViolence', 'Violence']

# Set confidence thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5
VIOLENCE_CONFIDENCE_THRESHOLD = 0.5

# Email Configuration
EMAIL_SENDER = "21071a6674@vnrvjiet.in"
# qvqv knzi wcun ehkd
EMAIL_PASSWORD = "qvqv knzi wcun ehkd" 
EMAIL_RECEIVER = "vattikuti.2004@gmail.com"

def send_email_with_attachment(image_path):
    """Sends an email with the captured violence detection frame as an attachment."""
    msg = EmailMessage()
    msg['Subject'] = 'Violence Detected Alert'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content("Violence has been detected. Please check the attached frame.")

    # Attach image
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        img_type = imghdr.what(image_path)  # Detect file type
        if img_type is None:
            img_type = "jpg"  # Default to JPG if detection fails
        img_name = f"violence_frame.{img_type}"  # Set file name dynamically
        msg.add_attachment(img_data, maintype='image', subtype=img_type, filename=img_name)

    # Send Email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print("Email sent successfully!")


def send_email_with_frame(frame):
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()
    
    msg = EmailMessage()
    msg['Subject'] = "Violence Alert!"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content("Violence detected! See the attached frame.")
    
    msg.add_attachment(img_bytes, maintype='image', subtype=imghdr.what(None, img_bytes), filename="violence_frame.jpg")
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_messxage(msg)

st.title("Violence Detection System")

# Upload file or live webcam
option = st.radio("Choose Input Source:", ("Upload Video", "Live Webcam"))

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv", "gif"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = None

elif option == "Live Webcam":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

if cap:
    stframe = st.empty()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform person detection
        person_results = person_model(frame)
        persons = []

        for result in person_results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0]
                if conf > PERSON_CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:  # Class '0' for 'person'
                    x1, y1, x2, y2 = box.xyxy[0]
                    persons.append((int(x1), int(y1), int(x2), int(y2)))

        # Perform violence detection if a person is detected
        if persons:
            violence_results = violenceDetect_model(frame)
            for result in violence_results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    if conf > VIOLENCE_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Draw rectangles based on the detection class
                        if currentClass == 'Violence':
                          for px1, py1, px2, py2 in persons:
                              if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                  cvzone.putTextRect(frame, f'Violence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                                  # Save frame
                                  frame_path = "violence_detected.jpg"
                                  cv2.imwrite(frame_path, frame)

                                  # Send email alert
                                  send_email_with_attachment(frame_path)
                                  break  # Send only one email per frame

                            # Check if the violence box overlaps with any person box
                            # for px1, py1, px2, py2 in persons:
                            #     if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            #         cvzone.putTextRect(frame, f'Violence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                            #         send_email_with_frame(frame)  # Send email with detected frame
                            #         break

                        elif currentClass == 'NonViolence':
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cvzone.putTextRect(frame, f'NonViolence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
