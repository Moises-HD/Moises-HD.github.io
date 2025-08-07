import streamlit as st
import json
import os

# Configuración
FICHERO_ETIQUETADO = "dataset_etiquetado.jsonl"

# Comprobar si existe
if not os.path.exists(FICHERO_ETIQUETADO):
    st.error("❌ No se ha encontrado el fichero de etiquetas. Primero debes etiquetar algunos registros.")
    st.stop()

# Cargar etiquetas
with open(FICHERO_ETIQUETADO, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

total = len(dataset)

if total == 0:
    st.warning("⚠️ El fichero de etiquetas está vacío.")
    st.stop()

# Navegación
indice = st.number_input("Selecciona el índice a revisar", min_value=1, max_value=total, value=1, step=1)

# Mostrar el registro seleccionado
registro = dataset[indice-1]

st.header(f"Revisión del ejemplo {indice} de {total}")

with st.expander("Ver OCR completo"):
    st.write(registro["ocr"])

st.subheader("Etiquetas guardadas:")
st.write(registro["labels"])
