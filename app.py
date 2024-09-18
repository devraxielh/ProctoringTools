import cv2
import streamlit as st

# Carga del clasificador preentrenado de OpenCV para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para detectar rostros en la imagen
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Función principal para la aplicación Streamlit
def main():
    st.title("Detección de una sola persona mirando la pantalla")

    # Activar la cámara
    run = st.checkbox('Activar Cámara')

    # Mostrar el feed de la cámara
    FRAME_WINDOW = st.image([])

    # Inicializar la cámara
    camera = cv2.VideoCapture(0)

    while run:
        # Leer el cuadro de la cámara
        ret, frame = camera.read()
        if not ret:
            st.write("Error al acceder a la cámara")
            break

        # Detección de rostros
        faces = detect_faces(frame)
        num_faces = len(faces)

        # Dibujar rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Mostrar el estado de detección
        if num_faces == 1:
            st.write("Una sola persona mirando la pantalla.")
        elif num_faces > 1:
            st.write(f"{num_faces} personas mirando la pantalla.")
        else:
            st.write("No hay personas mirando la pantalla.")

        # Mostrar la imagen en Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Liberar la cámara cuando termine
    camera.release()

if __name__ == "__main__":
    main()