import sys
import cv2
import face_recognition
import numpy as np
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QApplication
from PyQt5.QtGui import QPalette, QColor
import threading

# Load known faces and names
known_face_encodings = []
known_face_names = []

# Load images from the folder
images_folder = "images"
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(images_folder, filename)
        img = face_recognition.load_image_file(img_path)
        img_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(img_encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use the filename (without extension) as the name

# Initialize video capture
video_capture = cv2.VideoCapture(0)
recognized_name = "Unknown"
name_lock = threading.Lock()  # Create a lock for thread safety

class VideoCaptureThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(QtGui.QImage)

    def run(self):
        global recognized_name
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                current_recognized_name = "Unknown"

                # Use the known face with the smallest distance to the new face
                if matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        current_recognized_name = known_face_names[best_match_index]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, current_recognized_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Update the recognized name in a thread-safe manner
                with name_lock:
                    recognized_name = current_recognized_name
                    print(recognized_name)
            # Convert frame to QImage format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

            # Emit the new image
            self.change_pixmap_signal.emit(qt_image)

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the UI
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2e2e2e; border-radius: 10px;")

        # Create layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Video label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #4CAF50; border-radius: 10px;")
        self.layout.addWidget(self.video_label)

        # Thread for video capture
        self.thread = VideoCaptureThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def update_image(self, image):
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(image))

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(46, 46, 46))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(46, 46, 46))
    palette.setColor(QPalette.AlternateBase, QColor(46, 46, 46))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(46, 46, 46))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
