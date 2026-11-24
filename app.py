import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Детектор объектов (YOLO)", page_icon="📦")

# Заголовок приложения
st.title("📦 Детектор объектов (YOLO)")
st.write(
    "Загрузите фото или сделайте снимок с телефона — нейросеть найдёт и"
    " распознает объекты."
)


# Загружаем лёгкую модель YOLO (скачается автоматически при первом запуске)
@st.cache_resource
def load_model():
  return YOLO("yolov8n.pt")


with st.spinner("Загрузка нейросети..."):
  model = load_model()

# --- ОСНОВНАЯ ЧАСТЬ: Загрузка изображения ---
uploaded_file = st.file_uploader(
    "Выберите или сфотографируйте изображение...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Загруженное изображение", width="stretch")

  if st.button("🚀 Распознать объекты"):
    with st.spinner("Анализируем изображение..."):
      img_np = np.array(image)
      results = model(img_np)

      res_plotted = results[0].plot()
      res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

      st.subheader("Результат распознавания:")
      st.image(res_rgb, caption="Результат анализа", width="stretch")

      st.write("### Обнаруженные классы:")
      boxes = results[0].boxes
      if len(boxes) == 0:
        st.info("Объекты не обнаружены, попробуйте другое фото.")
      else:
        for box in boxes:
          cls_id = int(box.cls[0])
          class_name = model.names[cls_id]
          conf = float(box.conf[0]) * 100
          st.write(f"- **{class_name}** (уверенность: {conf:.1f}%)")