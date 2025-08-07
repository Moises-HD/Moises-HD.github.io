import sys
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os

# 🧪 Versión de demostración (limitada)
model = ocr_predictor(pretrained=True)

# NOTA: Este script está limitado para pruebas internas. Contactar para acceso completo al pipeline completo.


def procesar_pdf(pdf_path):
    try:
        doc = DocumentFile.from_pdf(pdf_path)
        result = model(doc)

        # Extraemos las líneas de texto OCR de todas las páginas
        ocr_lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text_line = " ".join([word.value for word in line.words])
                    ocr_lines.append(text_line)

        # Generamos el esqueleto del dataset para etiquetar manualmente
        sample = {
            "ocr": ocr_lines,
            "labels": {
                "fechaDesde": "",
                "fechaHasta": "",
                "cups": "",
                "consumo": "",
                "euroConsumo": "",
                "total": "",
                "excesoPotencia": ""
            }
        }

        return sample

    except Exception as e:
        print(f"Error procesando {pdf_path}: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python generar_dataset.py carpeta_pdfs/")
        sys.exit(1)

    carpeta = sys.argv[1]

    if not os.path.isdir(carpeta):
        print(f"La ruta {carpeta} no es una carpeta válida.")
        sys.exit(1)

    # Generamos el fichero de dataset
    with open("dataset.jsonl", "w", encoding="utf-8") as outfile:
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(".pdf"):
                pdf_path = os.path.join(carpeta, archivo)
                print(f"Procesando: {pdf_path}")
                muestra = procesar_pdf(pdf_path)
                if muestra:
                    outfile.write(json.dumps(muestra, ensure_ascii=False) + "\n")

    print("✅ Dataset generado: dataset.jsonl")
