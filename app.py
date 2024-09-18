import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Carga de clasificadores preentrenados de OpenCV para detección de rostros y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Procesador de video para detectar rostros y ojos
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        # Inicializar el número de rostros y si la persona está mirando a la cámara
        self.num_faces = 0
        self.looking_at_camera = False  # Asegurarse de inicializar este atributo

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convertir a escala de grises para la detección de rostros y ojos
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        self.num_faces = len(faces)

        # Asumimos que no están mirando a la cámara hasta verificar
        self.looking_at_camera = False

        # Dibujar rectángulos alrededor de los rostros detectados y verificar ojos
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Detectar los ojos dentro del rostro detectado
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(eyes) >= 2:  # Al menos dos ojos detectados
                self.looking_at_camera = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Devolver el frame procesado
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Función principal para la aplicación Streamlit
def main():
    st.title("Detección de persona mirando a la cámara")

    # Inicializar el streamer de video con la clase de procesamiento
    ctx = webrtc_streamer(key="example", video_processor_factory=FaceDetectionProcessor)

    # Mostrar el estado de detección de rostros y si están mirando a la cámara
    if ctx.video_processor:
        num_faces = ctx.video_processor.num_faces
        if num_faces == 1:
            if ctx.video_processor.looking_at_camera:
                st.write("Una persona está mirando a la cámara.")
            else:
                st.write("Una persona detectada, pero no está mirando a la cámara.")
        elif num_faces > 1:
            st.write(f"{num_faces} personas detectadas.")
        else:
            st.write("No hay personas mirando la pantalla.")

if __name__ == "__main__":
    main()