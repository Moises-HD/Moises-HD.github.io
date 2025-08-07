# inferir_factura.py
import sys, json, torch, re, os, tempfile, unicodedata
from datetime import datetime
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR = "./t5_factura_base_model"
CAMPOS = ["fechaDesde","fechaHasta","cups","consumo","euroConsumo","total","excesoPotencia"]

# =========================
# OCR (con fallback embebido)
# =========================
def ocr_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    embedded = []
    for page in doc:
        embedded.append(page.get_text("text"))
    embedded_text = "\n".join(s.strip() for s in embedded if s and s.strip())
    embedded_text = "\n".join(l.strip() for l in embedded_text.splitlines() if l.strip())
    if len(embedded_text) >= 50:
        if os.environ.get("DEBUG_INFER",""):
            open("debug_ocr.txt","w",encoding="utf-8").write(embedded_text)
            print("LEN_OCR_TEXT (embedded):", len(embedded_text), file=sys.stderr)
        return embedded_text

    # Si no hay texto embebido, rasterizamos y hacemos OCR
    zoom_candidates = [3.2, 3.8]
    for zoom in zoom_candidates:
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(zoom, zoom)
        tmpdir = tempfile.TemporaryDirectory()
        png_paths = []

        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            if pix.width < 32 or pix.height < 32:
                continue
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = ImageOps.grayscale(img)
            img = ImageOps.autocontrast(img)
            img = img.filter(ImageFilter.SHARPEN)

            max_side = 4500
            w, h = img.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

            out_path = os.path.join(tmpdir.name, f"page_{i+1:03d}.png")
            img.save(out_path, format="PNG")
            png_paths.append(out_path)

        if not png_paths:
            continue

        docfile = DocumentFile.from_images(png_paths)
        ocr = ocr_predictor(pretrained=True, assume_straight_pages=True)
        result = ocr(docfile)

        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    lines.append(" ".join(w.value for w in line.words))

        full_text = "\n".join(lines).replace("\xa0"," ").replace("\t"," ").strip()
        full_text = "\n".join(l.strip() for l in full_text.splitlines() if l.strip())

        if os.environ.get("DEBUG_INFER",""):
            open("debug_ocr.txt","w",encoding="utf-8").write(full_text)
            print(f"LEN_OCR_TEXT (zoom {zoom}):", len(full_text), file=sys.stderr)

        if len(full_text) >= 50:
            return full_text

    return ""

# =========================
# Parsing y normalización
# =========================
def extract_json_block(s: str) -> str:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        raise ValueError("No JSON found")
    return re.sub(r",\s*([}\]])", r"\1", m.group(0))

def normalize_out(out: dict) -> dict:
    # No tocar numéricos aquí; se normalizan en coerce_and_fill
    out["cups"] = (out.get("cups","") or "").replace(" ", "").upper()
    for k in ("fechaDesde","fechaHasta"):
        out[k] = (out.get(k,"") or "").replace(".", "/")
    return out

def parse_target(texto: str) -> dict:
    try:
        obj = json.loads(extract_json_block(texto))
        if isinstance(obj, dict):
            return normalize_out({k: obj.get(k,"") for k in CAMPOS})
    except Exception:
        pass
    out = {k:"" for k in CAMPOS}
    for parte in texto.split(";"):
        if ":" in parte:
            k, v = parte.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in out:
                out[k] = v
    return normalize_out(out)

# =========================
# Validadores y fallbacks
# =========================
CUPS_RE = re.compile(r"\bES[0-9A-Z]{18,24}\b")

def norm_date(s):
    s = (s or "").strip()
    if not s: return ""
    s = s.replace(".", "/").replace("-", "/")
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if not m: return ""
    d, mth, y = m.groups()
    y = int(y)
    if y < 100: y += 2000
    try: return datetime(y, int(mth), int(d)).strftime("%d/%m/%Y")
    except ValueError: return ""

def parse_number_keep_decimals(s: str) -> str:
    """
    Captura importes con posible signo: -67,93 | −67,93 | –67,93 | 1.648,41 | 1648.41 ...
    y normaliza a punto decimal.
    """
    if not s: return ""
    m = re.search(r"([\-−–]?\s*)(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\b", s)
    if not m: return ""
    sign, val = m.group(1), m.group(2)
    sign = "-" if sign.strip() in ("-", "−", "–") else ""
    val = val.replace(" ", "")
    if "." in val and "," in val:
        val = val.replace(".", "").replace(",", ".")
    else:
        val = val.replace(",", ".")
    return f"{sign}{val}"

def first_amount(s):  # más robusto, permite coma como separador decimal
    s = s.replace("−", "-").replace("–", "-")  # normaliza signos
    m = re.search(r"(-?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|-?\s*\d+(?:[.,]\d+)?)", s)
    if not m:
        return ""
    val = m.group(1).replace(" ", "")
    if "." in val and "," in val:
        val = val.replace(".", "").replace(",", ".")
    elif "," in val and "." not in val:
        val = val.replace(",", ".")
    return val


def find_consumo_kwh(ocr: str) -> str:
    for line in ocr.splitlines():
        low = line.lower().replace(" ", "")
        if "consumo" in low or "kwh" in low:
            n = parse_number_keep_decimals(line)
            if n: return n
    return ""

def find_total_kwh_from_table(ocr: str) -> str:
    # "Total 40.351 kWh" -> 40351
    m = re.search(r"Total\s+(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*kwh", ocr, flags=re.I)
    if not m:
        return ""
    val = m.group(1).replace(" ", "")
    if "." in val and "," in val:
        val = val.replace(".", "").replace(",", ".")
        try:
            f = float(val)
            return str(int(round(f))) if f.is_integer() else f"{f:.2f}"
        except: return val
    if "," not in val and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", val):
        return val.replace(".", "")
    if "," in val and "." not in val:
        val = val.replace(".", "").replace(",", ".")
        try:
            f = float(val)
            return str(int(round(f))) if f.is_integer() else f"{f:.2f}"
        except: return val
    try:
        f = float(val)
        return str(int(round(f))) if f.is_integer() else f"{f:.2f}"
    except: return val

def sum_termino_energia(ocr: str) -> str:
    lines = ocr.splitlines()
    total = 0.0

    for i, line in enumerate(lines):
        low = line.lower()

        if "energía activa" in low and "cargos" not in low and "acceso" not in low:

            # Intenta extraer importe de esta línea o siguientes
            found = False
            for offset in range(0, 6):  # busca en esta línea y las 5 siguientes
                if i + offset >= len(lines):
                    break
                next_line = lines[i + offset]
                amt = first_amount(next_line)
                if amt:
                    try:
                        val = float(amt)
                        if val > 50:  # ignora cantidades triviales
                            total += val
                            found = True
                            break
                    except:
                        continue

    return f"{total:.2f}" if abs(total) > 0.01 else ""


def find_cups_in_ocr(ocr):
    compact = ocr.replace(" ", "")
    m = CUPS_RE.search(compact)
    if m: return m.group(0)
    m = CUPS_RE.search(ocr)
    return m.group(0).replace(" ", "") if m else ""

def find_amount_after_labels(ocr, labels):
    """
    Busca un importe relacionado con ciertas etiquetas.
    Si no hay número en la misma línea, busca en las 3 siguientes.
    """
    lines = ocr.splitlines()
    for idx, line in enumerate(lines):
        low = line.lower()
        if any(lbl in low for lbl in labels):
            
            # Buscar en esta línea y hasta 3 siguientes
            for offset in range(0, 4):
                if idx + offset >= len(lines):
                    break
                current_line = lines[idx + offset]

                # 1. Número seguido de € o Eur
                m = re.search(r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?)(?:\s*(?:eur|€))", current_line, flags=re.I)
                if m:
                    return parse_number_keep_decimals(m.group(0))

                # 2. Último número de la línea
                nums = re.findall(r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\-?\s*\d+(?:[.,]\d+)?)", current_line)
                if nums:
                    return parse_number_keep_decimals(nums[-1])
    return ""

def norm_text(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def find_total_factura_strict(ocr: str) -> str:
    """
    Encuentra el importe de 'Total factura' aunque esté en la línea siguiente.
    Estrategia: buscar 'total factura' en el texto normalizado y, desde ahí,
    escanear los siguientes 200 caracteres para el primer número (con signo),
    preferentemente seguido de '€' o 'Eur'.
    """
    flat = norm_text(ocr)
    # normaliza espacios y baja a minúsculas para detectar el ancla
    flat_compact = re.sub(r"\s+", " ", flat)
    lower = flat_compact.lower()

    # ancla
    m = re.search(r"\btotal\s*factura\b", lower)
    if not m:
        return ""

    start = m.end()  # justo después de 'total factura'
    # recorta una ventana razonable por si el importe está en la línea siguiente
    window = flat_compact[start:start+200]

    # 1) número (con signo) antes de EUR/€
    m_eur = re.search(
        r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\-?\s*\d+(?:[.,]\d+)?)(?:\s*(?:eur|€))",
        window, flags=re.I
    )
    if m_eur:
        return parse_number_keep_decimals(m_eur.group(0))

    # 2) si no hay 'eur/€', toma el PRIMER número de la ventana
    m_num = re.search(
        r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\-?\s*\d+(?:[.,]\d+)?)",
        window
    )
    if m_num:
        return parse_number_keep_decimals(m_num.group(0))

    return ""


def extract_period_dates(ocr: str):
    m = re.search(r"PERIODO\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*[-–]\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", ocr, flags=re.I)
    if not m:
        return "", ""
    d1, d2 = m.group(1), m.group(2)
    return norm_date(d1), norm_date(d2)

def sum_euro_consumo_por_periodo(ocr: str) -> str:
    """
    Suma importes de energía activa por periodos (P1...P6).
    Ejemplo:
        Término energía P1 123,45 €
        Término energía P2 456,78 €
        ...
    """
    total = 0.0
    lineas_utiles = []

    for line in ocr.splitlines():
        lower = line.lower()
        if "término energía" in lower or "termino energia" in lower or "energía activa" in lower:
            if "cargos" in lower or "acceso" in lower:
                continue  # Ignora líneas no deseadas
            lineas_utiles.append(line)

    for linea in lineas_utiles:
        num = parse_number_keep_decimals(linea)
        if num:
            try:
                total += float(num)
            except:
                continue

    return f"{total:.2f}" if total > 0.01 else ""


def coerce_and_fill(out, ocr_text):
    out["euroConsumo"] = ""  # ← 🔧 RESET

    euro_final = ""
    # Fechas
    out["fechaDesde"] = norm_date(out.get("fechaDesde",""))
    out["fechaHasta"] = norm_date(out.get("fechaHasta",""))

    # CUPS
    cups = out.get("cups","").replace(" ", "").upper()
    if not CUPS_RE.fullmatch(cups):
        cups_fallback = find_cups_in_ocr(ocr_text) or ""
        out["cups"] = cups_fallback or cups

    # Consumo: modelo -> fallback genérico -> tabla Total kWh (override)
    v = out.get("consumo","")
    if re.match(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", v or ""): v = ""
    out["consumo"] = parse_number_keep_decimals(v) if v else ""
    if not out["consumo"]:
        out["consumo"] = find_consumo_kwh(ocr_text)
    table_kwh = find_total_kwh_from_table(ocr_text)
    if table_kwh:
        out["consumo"] = table_kwh

    # euroConsumo: siempre sobrescribir con lógica OCR
    euro_final = ""

    # 1. Buscar por etiquetas
    euro_etiqueta = find_amount_after_labels(
        ocr_text,
        ["término energía", "termino energia", "importe energía", "energia activa", "consumo energía"]
    )
    if euro_etiqueta:
        try:
            if float(euro_etiqueta) > 50:
                euro_final = euro_etiqueta
            
        except:
            pass


    # 2. Sumar bloques relevantes
    if not euro_final:
        euro_sum = sum_termino_energia(ocr_text)
        if euro_sum:
            euro_final = euro_sum

    # 3. Más preciso: por periodo (solo si es más alto)
    euro_por_periodo = sum_euro_consumo_por_periodo(ocr_text)
    try:
        if euro_por_periodo and float(euro_por_periodo) > float(euro_final or "0") and float(euro_por_periodo) > 50:
            euro_final = euro_por_periodo
    except:
        pass



    # Anular en rectificativas
    if "rectifica" in ocr_text.lower() or "rectificativa" in ocr_text.lower():
        euro_final = ""
        out["excesoPotencia"] = ""
        out["consumo"] = ""

    # Siempre sobrescribir
    out["euroConsumo"] = euro_final

    # total: SOLO "Total factura ..." o variantes; sin fallback de "último €"
    tf = find_total_factura_strict(ocr_text)
    if tf:
        out["total"] = tf
    else:
        label_total = find_amount_after_labels(
            ocr_text,
            ["total a pagar", "importe total", "total a abonar", "totalfactura"]
        )
        if label_total:
            out["total"] = parse_number_keep_decimals(label_total)

    # Normalizar predicción inicial
    pred_exceso = out.get("excesoPotencia", "")
    pred_exceso_valido = ""

    try:
        if pred_exceso:
            val = float(pred_exceso.replace(",", "."))
            if val > 10:
                # Validar que aparece junto a etiquetas válidas
                found_valid_label = False
                for line in ocr_text.splitlines():
                    lower = line.lower()
                    if any(lbl in lower for lbl in ["excesos de potencia", "exceso potencia", "exceso de potencia"]):
                        if pred_exceso in line or pred_exceso.replace(".", ",") in line:
                            found_valid_label = True
                            break
                if found_valid_label:
                    pred_exceso_valido = pred_exceso
    except:
        pass

    # Fallback si sigue vacío
    if not pred_exceso_valido:
        exceso = find_amount_after_labels(
            ocr_text,
            ["excesos de potencia", "exceso potencia", "exceso de potencia"]
        )
        try:
            if exceso and float(exceso.replace(",", ".")) > 10:
                pred_exceso_valido = exceso
        except:
            pass

    out["excesoPotencia"] = parse_number_keep_decimals(pred_exceso_valido) if pred_exceso_valido else ""

    # excesoPotencia (si algún día viene en €)
    out["excesoPotencia"] = parse_number_keep_decimals(out.get("excesoPotencia","")) if out.get("excesoPotencia") else ""

    # Fechas desde PERIODO (override final)
    d1, d2 = extract_period_dates(ocr_text)
    if d1: out["fechaDesde"] = d1
    if d2: out["fechaHasta"] = d2

    return out

# =========================
# Prompt & T5
# =========================
def build_prompt(ocr_text: str, max_chars=8000) -> str:
    instrucciones = (
        "Extrae estos campos de una factura eléctrica y responde SOLO en el formato "
        "'clave:valor; ...' usando EXACTAMENTE estas claves: "
        f"{', '.join(CAMPOS)}. No inventes valores."
    )
    texto = ocr_text[:max_chars]
    return f"{instrucciones}\n\nTEXTO_OCR:\n{texto}"

def infer_from_text(ocr_text, max_input=1536, max_new_tokens=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    prompt = build_prompt(ocr_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input).to(device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4,
                         length_penalty=0.8, early_stopping=True)
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    if os.environ.get("DEBUG_INFER",""):
        open("debug_t5.txt","w",encoding="utf-8").write(decoded)
        print("RAW_T5_OUTPUT:", decoded, file=sys.stderr)
    out = parse_target(decoded)
    out = coerce_and_fill(out, ocr_text)
    return out

def evaluar_accuracy(predicho, esperado):
    campos = CAMPOS
    aciertos = {}
    total = len(campos)
    correctos = 0

    for campo in campos:
        v_pred = (predicho.get(campo) or "").strip()
        v_real = (esperado.get(campo) or "").strip()

        es_igual = v_pred == v_real

        # 🔍 Extra: comparar como números si parece numérico
        try:
            f_pred = float(v_pred.replace(",", "."))
            f_real = float(v_real.replace(",", "."))
            if abs(f_pred - f_real) < 0.01:
                es_igual = True
        except:
            pass

        aciertos[campo] = {"predicho": v_pred, "esperado": v_real, "ok": es_igual}
        if es_igual:
            correctos += 1

    acc_global = correctos / total
    return aciertos, acc_global


# =========================
# Main
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error":"Uso: python inferir_factura.py factura.pdf"}, ensure_ascii=False))
        sys.exit(1)
    pdf_path = sys.argv[1]
    try:
        ocr_text = ocr_pdf_to_text(pdf_path)
        if len(ocr_text) == 0:
            print(json.dumps({"error":"OCR vacío"}, ensure_ascii=False))
            sys.exit(2)
        resultado = infer_from_text(ocr_text)
        print(json.dumps(resultado, ensure_ascii=False))

        # ==== EVALUACIÓN (opcional, con valores esperados hardcoded) ====
        esperado = {
            "fechaDesde": "01/05/2025",
            "fechaHasta": "31/05/2025",
            "cups": "ES0000000000000000XXXX",
            "consumo": "37703",
            "euroConsumo": "2987.4",
            "total": "4888.42",
            "excesoPotencia": "24.29"
        }

        aciertos, acc = evaluar_accuracy(resultado, esperado)
        print("\n== EVALUACIÓN ==")
        for campo, detalle in aciertos.items():
            estado = "✅" if detalle["ok"] else "❌"
            print(f"{estado} {campo}: predicho='{detalle['predicho']}' / esperado='{detalle['esperado']}'")
        print(f"\n📊 Accuracy total: {acc:.2%}")

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

