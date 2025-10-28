import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys
from PIL import Image

# 🎧 Configuración de la página con estilo "Real Hasta La Muerte"
st.set_page_config(
    page_title="🎤 Detector Real Hasta La Muerte",
    page_icon="💀",
    layout="wide"
)

# Imagen decorativa
header_img = Image.open("anuel4.png")  # Puedes usar un logo de Anuel
st.image(header_img, width=300)

# Título principal
st.title("💀 Detector Real Hasta La Muerte 🔥")
st.markdown("""
Bienvenido al **detector urbano de Anuel**, una app que usa **YOLOv5** para reconocer objetos como si estuvieras  
grabando un video musical o revisando lo que hay en tu entorno.

👁️ **Detecta lo que te rodea** y mira cómo el modelo identifica los objetos con precisión.  
Inspirado en la visión de *Anuel AA*, donde nada pasa desapercibido.  
""")

# --- Cargar el modelo YOLOv5 ---
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning("⚙️ Intentando método alternativo de carga...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        🧠 Sugerencias:
        - Instala una versión compatible de PyTorch y YOLOv5  
        - Asegúrate de tener el archivo del modelo `yolov5s.pt`
        - Si no, se puede descargar automáticamente desde Torch Hub
        """)
        return None

with st.spinner("🎶 Cargando el modelo de detección..."):
    model = load_yolov5_model()

# --- Si el modelo se cargó correctamente ---
if model:
    st.sidebar.title("⚙️ Ajustes de detección")
    st.sidebar.markdown("Personaliza la visión del detector de Anuel 👁️‍🗨️")

    with st.sidebar:
        st.subheader("🎚️ Sensibilidad")
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

        st.subheader("🎛️ Opciones Avanzadas")
        try:
            model.agnostic = st.checkbox('NMS sin clases', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("⚠️ Algunas opciones no están disponibles en esta versión.")

    # --- Detección ---
    st.markdown("## 📸 Captura tu entorno y deja que Anuel vea lo que tú ves")
    picture = st.camera_input("Pulsa para tomar una foto 🎥")

    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("🔍 Analizando la escena..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"💥 Error durante la detección: {str(e)}")
                st.stop()

        try:
            predictions = results.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🖼️ Resultado visual")
                results.render()
                st.image(cv2_img, channels='BGR', use_container_width=True)
                st.caption("🎨 Detecciones procesadas con el estilo Real Hasta La Muerte")

            with col2:
                st.subheader("📋 Objetos detectados")
                label_names = model.names
                category_count = {}

                for category in categories:
                    category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[category_idx] = category_count.get(category_idx, 0) + 1

                data = []
                for category, count in category_count.items():
                    label = label_names[category]
                    confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                    data.append({
                        "Categoría": label,
                        "Cantidad": count,
                        "Confianza promedio": f"{confidence:.2f}"
                    })

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index('Categoría')['Cantidad'])
                else:
                    st.info("😶 No se detectaron objetos con la configuración actual.")
                    st.caption("📉 Intenta reducir el umbral de confianza en la barra lateral.")

        except Exception as e:
            st.error(f"Error procesando resultados: {str(e)}")
            st.stop()

else:
    st.error("🚫 No se pudo cargar el modelo YOLOv5. Revisa las dependencias e inténtalo otra vez.")
    st.stop()

# --- Pie de página ---
st.markdown("---")
st.caption("""
👑 **Detector Real Hasta La Muerte** — YOLOv5 x Streamlit  
Analiza objetos en tiempo real con el estilo y la visión de *Anuel AA*.  
Hecho por fans, para fans. 🔥
""")
