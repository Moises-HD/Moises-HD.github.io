import streamlit as st
import json
import os

DATASET_PATH = 'dataset_final.jsonl'
OUTPUT_PATH = 'dataset_etiquetado.jsonl'

# Cargar el dataset
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Cargar el progreso si existe
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        etiquetas = [json.loads(line) for line in f]
else:
    etiquetas = []

# Determinar el índice actual
indice_actual = len(etiquetas)
total_facturas = len(dataset)

if indice_actual >= total_facturas:
    st.success("🎉 ¡Ya has terminado de etiquetar todas las facturas!")
else:
    muestra = dataset[indice_actual]

    # Mostrar progreso arriba
    st.title("🧾 Etiquetador de Facturas OCR")

    progreso_texto = f"Factura {indice_actual + 1} de {total_facturas}"
    st.write(progreso_texto)

    progreso_porcentaje = (indice_actual + 1) / total_facturas
    st.progress(progreso_porcentaje)

    # Mostrar todo el OCR como un único texto grande
    ocr_texto = "\n".join(muestra['ocr'])
    st.text_area("OCR completo de la factura", ocr_texto, height=400)

    st.header("Introduce las etiquetas:")

    # Campos a etiquetar
    fechaDesde = st.text_input("Fecha Desde", value=muestra['labels'].get('fechaDesde', ''))
    fechaHasta = st.text_input("Fecha Hasta", value=muestra['labels'].get('fechaHasta', ''))
    cups = st.text_input("CUPS", value=muestra['labels'].get('cups', ''))
    consumo = st.text_input("Consumo (kWh)", value=muestra['labels'].get('consumo', ''))
    euroConsumo = st.text_input("Consumo (€)", value=muestra['labels'].get('euroConsumo', ''))
    total = st.text_input("Total (€)", value=muestra['labels'].get('total', ''))
    excesoPotencia = st.text_input("Exceso Potencia (€)", value=muestra['labels'].get('excesoPotencia', ''))

    if st.button("💾 Guardar y siguiente"):
        muestra['labels'] = {
            'fechaDesde': fechaDesde,
            'fechaHasta': fechaHasta,
            'cups': cups,
            'consumo': consumo,
            'euroConsumo': euroConsumo,
            'total': total,
            'excesoPotencia': excesoPotencia
        }
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(muestra, ensure_ascii=False) + '\n')
        st.rerun()
