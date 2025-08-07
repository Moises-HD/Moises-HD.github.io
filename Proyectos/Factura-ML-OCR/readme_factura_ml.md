# Proyecto: Extracción de Datos de Facturas Eléctricas con OCR + T5

Este proyecto aplica una combinación de OCR (Reconocimiento Óptico de Caracteres) y un modelo de lenguaje (T5) para extraer automáticamente campos clave de facturas eléctricas en PDF.

## 📈 Objetivo

Extraer información estructurada a partir de facturas energéticas:

- Fecha de inicio y fin del periodo facturado
- CUPS
- Consumo (kWh)
- Importe por energía activa
- Total de la factura
- Exceso de potencia facturado (si aplica)

---

## 🤖 Tecnologías utilizadas

- [Doctr](https://mindee.github.io/doctr/) para OCR
- [Transformers (HuggingFace)](https://huggingface.co/docs/transformers) para modelado con `t5-base`
- `PyMuPDF` (`fitz`) para lectura de PDFs
- `Streamlit` para la interfaz de etiquetado y revisión

---

## 📂 Estructura de Carpetas

```
Factura-ML-OCR/
├── generar_dataset.py            # OCR masivo de PDFs
├── etiquetador.py	          # Interfaz para etiquetar datos OCR
├── verificador.py	          # Comprobador de etiquetas
├── script_union.py               # Une dataset1.jsonl, dataset2.jsonl...
├── train_t5_base.py              # Entrenamiento del modelo T5
├── inferir_factura.py            # Infiere datos desde un PDF
├── dataset_final.jsonl           # (IGNORAR) Dataset completo
├── dataset_etiquetado.jsonl      # (IGNORAR) Etiquetas reales
├── t5_factura_base_model/        # (IGNORAR) Modelo entrenado
```

> ⚠️ **Nota**: Algunos archivos como el dataset y modelo entrenado **no están incluidos en el repositorio** por contener datos reales o ser muy pesados. Puedes entrenar tu propio modelo siguiendo las instrucciones.

---

🧪 Resultados: Capturas de ejemplo

Se incluyen capturas con ejemplos reales de ejecución donde el modelo ha obtenido hasta un 100% de precisión en campos clave extraídos de facturas.

📷 Accede a las capturas aquí: [capturas/](https://github.com/Moises-HD/Moises-HD.github.io/tree/main/Proyectos/Factura-ML-OCR/capturas)

Ejemplos:

✅ Precisión total del 100% con fechas, consumo, total, exceso de potencia y CUPS extraídos correctamente

⚠️ Casos con 85% de precisión también están incluidos como ejemplo de escenarios mejorables

Estos resultados demuestran la eficacia del sistema entrenado sobre facturas reales.

---

## ⚖️ 1. Generar Dataset OCR

```bash
python generar_dataset.py ./carpeta_pdfs
```

Esto crea un archivo `dataset.jsonl` con texto OCR y campos vacíos para etiquetar.

---

## 📅 2. Etiquetar datos con Streamlit

```bash
streamlit run etiquetador.py
```

Te permite etiquetar fácilmente los campos clave de cada factura y guarda el progreso.

---

## 🔍 3. Revisar etiquetas guardadas

```bash
streamlit run verificador.py
```

Visualiza cualquier entrada etiquetada para asegurarte de su calidad.

---

## ➕ 4. Unir datasets parciales

```bash
python script_union.py
```

Une `dataset1.jsonl`, `dataset2.jsonl`... en `dataset_final.jsonl` para entrenamiento.

---

## 🎓 5. Entrenar el modelo T5

```bash
python train_t5_base.py
```

Esto entrena un modelo T5 para inferencia posterior.

---

## ⚖️ 6. Inferir campos desde una factura PDF

```bash
python inferir_factura.py factura.pdf
```

Salida esperada:

```json
{
  "fechaDesde": "01/01/2025",
  "fechaHasta": "31/01/2025",
  "cups": "ESXXXXXXXXXXXXXXX",
  "consumo": "40904",
  "euroConsumo": "3768.25",
  "total": "6929.95",
  "excesoPotencia": "361.98"
}
```
---

🖼️ Ejemplos visuales de inferencia

Las siguientes capturas muestran salidas del sistema (accuracy parcial o total):

![capturas/Factura1.png](https://github.com/Moises-HD/Moises-HD.github.io/blob/main/Proyectos/Factura-ML-OCR/capturas/Factura1.png)

![capturas/Factura2.png](https://github.com/Moises-HD/Moises-HD.github.io/blob/main/Proyectos/Factura-ML-OCR/capturas/Factura2.png)

![capturas/Factura3.png](https://github.com/Moises-HD/Moises-HD.github.io/blob/main/Proyectos/Factura-ML-OCR/capturas/Factura3.png)

![capturas/Factura4.png](https://github.com/Moises-HD/Moises-HD.github.io/blob/main/Proyectos/Factura-ML-OCR/capturas/Factura4.png)

![capturas/Factura5.png](https://github.com/Moises-HD/Moises-HD.github.io/blob/main/Proyectos/Factura-ML-OCR/capturas/Factura5.png)

![capturas/Factura6.png](https://github.com/Moises-HD/Moises-HD.github.io/blob/main/Proyectos/Factura-ML-OCR/capturas/Factura6.png)

![capturas/Factura7.png](https://github.com/Moises-HD/Moises-HD.github.io/blob/main/Proyectos/Factura-ML-OCR/capturas/Factura7.png)

![capturas/Factura8.png](https://github.com/Moises-HD/Moises-HD.github.io/blob/main/Proyectos/Factura-ML-OCR/capturas/Factura8.png)

🔐 Todos los CUPS han sido anonimizados en las imágenes para proteger los datos personales.


---

## 🔐 Privacidad y Datos

Este proyecto **no incluye datos reales**. Todos los archivos `.jsonl` y modelos entrenados están excluidos con `.gitignore`.

Si deseas probar el sistema, genera tus propios PDFs o usa documentos ficticios.

---

## 🚀 Próximos pasos (uso empresarial)

- Evaluación masiva sobre test set (privada)
- Deploy con API REST o interfaz web (bajo NDA)
- Integración en portal empresarial (uso privado)

> ⚠️ Esta parte no se publica por motivos de confidencialidad.

---

## ✍️ Autor

Desarrollado por Moisés Herrada Díaz. Para más información o colaboraciones, contacta por GitHub o email.

---

## 📦 Licencia

Este repositorio está disponible bajo la licencia MIT. Puedes usar el código para fines educativos o profesionales con atribución adecuada.

---

🔗 Enlaces

🌐 Página del proyecto: https://moises-hd.github.io/Proyectos/Factura-ML-OCR/

📁 Ver código en GitHub: https://github.com/Moises-HD/Moises-HD.github.io/tree/main/Proyectos/Factura-ML-OCR