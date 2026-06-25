import cv2
import numpy as np
import streamlit as st
from PIL import Image

from mpi import tiene_pastilla


st.set_page_config(
    page_title="Detección de pastillas",
    page_icon="💊",
    layout="centered",
)

st.title("💊 Detección automática de pastillas faltantes")
st.write(
    "Sube una imagen del blíster para detectar automáticamente cuántas "
    "pastillas están presentes y cuántas faltan."
)

uploaded_file = st.file_uploader(
    "Sube una imagen del blíster",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = np.array(image)[:, :, ::-1]

    st.image(image, caption="Imagen original", use_container_width=True)

    target_width = 600
    scale_ratio = target_width / img_cv.shape[1]
    new_dimensions = (
        target_width,
        max(1, int(img_cv.shape[0] * scale_ratio)),
    )
    img_cv_resized = cv2.resize(img_cv, new_dimensions)

    gray = cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=35,
        param1=30,
        param2=25,
        minRadius=16,
        maxRadius=23,
    )

    result = img_cv_resized.copy()
    missing = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        total = circles.shape[1]

        for circle in circles[0, :]:
            x, y, radius = (int(value) for value in circle)

            x1 = max(x - radius, 0)
            y1 = max(y - radius, 0)
            x2 = min(x + radius, gray.shape[1])
            y2 = min(y + radius, gray.shape[0])
            patch = gray[y1:y2, x1:x2]

            if patch.size == 0:
                continue

            if tiene_pastilla(patch):
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
                missing += 1

            cv2.circle(result, (x, y), radius, color, 3)
            cv2.circle(result, (x, y), 2, (255, 0, 0), 3)

        st.write(f"🔎 Total de posiciones detectadas: **{total}**")
        st.write(f"❌ Pastillas faltantes: **{missing}**")

        if missing > 0:
            st.error("⚠️ Se encontraron pastillas faltantes.")
        else:
            st.success("✅ Todas las pastillas están presentes.")
    else:
        st.warning(
            "⚠️ No se detectaron círculos. Prueba con otra imagen o mejora "
            "la iluminación y el contraste."
        )

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Resultado", use_container_width=True)
