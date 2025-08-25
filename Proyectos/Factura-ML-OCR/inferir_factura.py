import sys, json, torch, re, os, tempfile, unicodedata
from datetime import datetime
from PIL import Image, ImageOps, ImageFilter
import fitz  
import time  

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
    Captura importes con posible signo: -67,93 | 1.648,41 | 1648.41 ...
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

def first_amount(s):  
    s = s.replace("−", "-").replace("–", "-") 
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

def find_euro_total_energia_line(ocr: str) -> str:
    """
    Detecta 'Total <NUM> kWh ... [importe]' aunque haya salto de línea,
    obligando a que tras 'Total' venga un número + 'kWh' (para no confundir con 'TOTAL IMPORTE ...').
    Devuelve el importe normalizado.
    """
    txt = (ocr.replace("\xa0"," ").replace("\u202f"," ").replace("\u2009"," ").replace("\u2007"," "))
    # 1) Encuentra el patrón 'Total <número> kWh' (esa parte sí debe ir en la misma "franja")
    m = re.search(r"\bTotal\s+(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+))\s*kWh\b", txt, flags=re.I)
    if not m:
        return ""
    end_pos = m.end()
    # 2) Desde ahí, busca el PRIMER importe (con o sin símbolo) en las ~5 líneas siguientes
    tail = txt[end_pos:end_pos+1200]  # ventana amplia pero acotada
    # preferimos el que lleve €/Eur
    m2 = re.search(r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+))\s*(?:€|Eur)\b", tail, flags=re.I)
    if m2:
        return parse_number_keep_decimals(m2.group(1))
    # si no hay €, tomamos el primer número “con pinta” de importe
    m3 = re.search(r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+))", tail)
    if m3:
        return parse_number_keep_decimals(m3.group(1))
    return ""

def sum_energia_consumida_iberdrola(ocr: str) -> str:
    txt = (ocr.replace("\xa0"," ").replace("\u202f"," ").replace("\u2009"," ").replace("\u2007"," "))
    total = 0.0
    # Captura 'Energía consumida P<dígitos> ... <importe> €'
    for line in txt.splitlines():
        if "energía consumida" in line.lower():
            m = re.search(r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+))\s*(?:€|Eur)\b", line, flags=re.I)
            if m:
                try:
                    total += float(parse_number_keep_decimals(m.group(1)))
                except:
                    pass
    return f"{total:.2f}" if total > 0.01 else ""


def find_euro_total_kwh_nearby(ocr: str) -> str:
    lines = (
        ocr.replace("\xa0", " ")
           .replace("\u202f", " ")
           .replace("\u2009", " ")
           .replace("\u2007", " ")
           .splitlines()
    )

    for i, line in enumerate(lines):
        low = line.lower()
        if "total" in low and "kwh" in low:
            window = lines[i : min(i + 6, len(lines))]

            # 1) Primero intenta con símbolo €
            for s in window:
                m = re.search(r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+))\s*(?:€|Eur)\b", s, flags=re.I)
                if m:
                    return parse_number_keep_decimals(m.group(1))

            # 2) Si no hay €, toma el ÚLTIMO número en la ventana que parezca importe (>50)
            cand = None
            for s in window:
                nums = re.findall(r"([\-−–]?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)|\d+(?:[.,]\d+)?)", s)
                if nums:
                    cand = nums[-1]  # último número encontrado
            if cand:
                val = parse_number_keep_decimals(cand)
                try:
                    f = float(val)
                    if f > 50:
                        return val
                except:
                    pass
            return ""  
    return ""

def _fmt_secs(s):
    # devuelve "Xm Ys" o "Z.ms" si es corto
    if s < 1:
        return f"{s*1000:.0f} ms"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h} h {m} min {sec} s"
    if m: return f"{m} min {sec} s"
    return f"{sec} s"

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

    # --- Saneamos espacios raros del OCR para que los regex no fallen ---
    ocr0 = (
        ocr_text.replace("\xa0", " ")
                .replace("\u202f", " ")
                .replace("\u2009", " ")
                .replace("\u2007", " ")
    )

    # =========================
    # euroConsumo (bloque ÚNICO)
    # =========================
    euro_final = ""

    # 0) "Total ... kWh ... [importe]" (soporta salto de línea)
    euro_total_line = find_euro_total_energia_line(ocr0)

    if euro_total_line:
        try:
            if float(euro_total_line) > 50:
                euro_final = euro_total_line
        except:
            pass

    # 0b) Fallback cercano: línea con 'Total' y 'kWh' + importe en siguientes líneas
    if not euro_final:
        euro_near = find_euro_total_kwh_nearby(ocr0)
        if euro_near:
            try:
                if float(euro_near) > 50:
                    euro_final = euro_near
            except:
                pass

    # 0c) Si sigue vacío, sumar explícitamente 'Energía consumida Pn ... €'
    if not euro_final:
        euro_sum_iber = sum_energia_consumida_iberdrola(ocr0)
        try:
            if euro_sum_iber and float(euro_sum_iber) > 50:
                euro_final = euro_sum_iber
        except:
            pass


    # 1) Si aún vacío, buscar por etiquetas típicas
    if not euro_final:
        euro_etiqueta = find_amount_after_labels(
            ocr0,
            ["término energía", "termino energia", "importe energía", "energia activa", "consumo energía"]
        )
        if euro_etiqueta:
            try:
                if float(euro_etiqueta) > 50:
                    euro_final = euro_etiqueta
            except:
                pass

    # 2) Si aún vacío, sumar bloques relevantes de "energía activa"
    if not euro_final:
        euro_sum = sum_termino_energia(ocr0)
        if euro_sum:
            euro_final = euro_sum

    # 3) Por periodo (si supera y es > 50)
    euro_por_periodo = sum_euro_consumo_por_periodo(ocr0)
    try:
        if euro_por_periodo and float(euro_por_periodo) > float(euro_final or "0") and float(euro_por_periodo) > 50:
            euro_final = euro_por_periodo
    except:
        pass

    # 4) Anular en rectificativas
    if "rectifica" in ocr0.lower() or "rectificativa" in ocr0.lower():
        euro_final = ""
        out["excesoPotencia"] = ""
        out["consumo"] = ""

    # Asigna euroConsumo
    out["euroConsumo"] = euro_final

    # =========================
    # Resto de campos
    # =========================

    # Fechas (normalización básica)
    out["fechaDesde"] = norm_date(out.get("fechaDesde",""))
    out["fechaHasta"] = norm_date(out.get("fechaHasta",""))

    # CUPS
    cups = (out.get("cups","") or "").replace(" ", "").upper()
    if not CUPS_RE.fullmatch(cups):
        cups_fallback = find_cups_in_ocr(ocr0) or ""
        out["cups"] = cups_fallback or cups
    else:
        out["cups"] = cups

    # Consumo: modelo -> fallback genérico -> tabla Total kWh (override)
    v = out.get("consumo","")
    if re.match(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", v or ""):
        v = ""
    out["consumo"] = parse_number_keep_decimals(v) if v else ""
    if not out["consumo"]:
        out["consumo"] = find_consumo_kwh(ocr0)
    table_kwh = find_total_kwh_from_table(ocr0)
    if table_kwh:
        out["consumo"] = table_kwh

    # total: SOLO "Total factura ..." o variantes; sin fallback de "último €"
    tf = find_total_factura_strict(ocr0)
    if tf:
        out["total"] = tf
    else:
        label_total = find_amount_after_labels(
            ocr0,
            ["total a pagar", "importe total", "total a abonar", "totalfactura"]
        )
        if label_total:
            out["total"] = parse_number_keep_decimals(label_total)

    # excesoPotencia (validación + fallback)
    pred_exceso = out.get("excesoPotencia", "")
    pred_exceso_valido = ""
    try:
        if pred_exceso:
            val = float(pred_exceso.replace(",", "."))
            if val > 10:
                # Validar que aparece junto a etiquetas válidas
                found_valid_label = False
                for line in ocr0.splitlines():
                    lower = line.lower()
                    if any(lbl in lower for lbl in ["excesos de potencia", "exceso potencia", "exceso de potencia"]):
                        if pred_exceso in line or pred_exceso.replace(".", ",") in line:
                            found_valid_label = True
                            break
                if found_valid_label:
                    pred_exceso_valido = pred_exceso
    except:
        pass

    if not pred_exceso_valido:
        exceso = find_amount_after_labels(
            ocr0,
            ["excesos de potencia", "exceso potencia", "exceso de potencia"]
        )
        try:
            if exceso and float(exceso.replace(",", ".")) > 10:
                pred_exceso_valido = exceso
        except:
            pass

    out["excesoPotencia"] = parse_number_keep_decimals(pred_exceso_valido) if pred_exceso_valido else ""
    if out["excesoPotencia"]:
        out["excesoPotencia"] = parse_number_keep_decimals(out["excesoPotencia"])

    # Fechas desde PERIODO (override final si aparecen)
    d1, d2 = extract_period_dates(ocr0)
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

def cargar_esperados_de_json(carpeta_o_pdf):
    """
    Si existe expected.json en la carpeta (o en la carpeta del PDF), lo carga.
    Formato esperado:
    {
      "factura10.pdf": {"fechaDesde": "...", ...},
      "cualquier.pdf": {...}
    }
    """
    ruta = carpeta_o_pdf
    if os.path.isfile(ruta):
        ruta = os.path.dirname(os.path.abspath(ruta))
    expected_path = os.path.join(ruta, "expected.json")
    if os.path.exists(expected_path):
        try:
            with open(expected_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] No se pudo leer expected.json: {e}")
    return {}

def esperado_por_nombre(nombre_pdf):
    """
    Fallback: si el nombre del archivo coincide con tus 'esperado_facturaX',
    devuélvelo. Ajusta/añade aquí lo que ya tienes en tu main.
    """
    base = nombre_pdf.lower()
    mapping = {
        "factura1.pdf": {
            "fechaDesde": "01/01/2025","fechaHasta": "31/01/2025",
            "cups": "","consumo": "",
            "euroConsumo": "","total": "","excesoPotencia": ""
        },
        "factura2.pdf": {
            
        },
        "factura3.pdf": {
            
        },
        "factura4.pdf": {
            
        },
        "factura5.pdf": {
            
        },
        "factura6.pdf": {
            
        },
        "factura7.pdf": {
            
        },
        "factura8.pdf": {
            
        },
        "factura9.pdf": {
            
        },
        "factura10.pdf": {
            
        },
        "factura11.pdf": {
            
        },
    }
    return mapping.get(base, None)

def evaluar_y_mostrar(resultado, esperado, titulo=""):
    aciertos, acc = evaluar_accuracy(resultado, esperado)
    print("\n== EVALUACIÓN ==" + (f" [{titulo}]" if titulo else ""))
    for campo, detalle in aciertos.items():
        estado = "✅" if detalle["ok"] else "❌"
        print(f"{estado} {campo}: predicho='{detalle['predicho']}' / esperado='{detalle['esperado']}'")
    print(f"📊 Accuracy: {acc:.2%}")
    return acc


# =========================
# Main
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error":"Uso: python inferir_factura.py <pdf_o_carpeta>"}, ensure_ascii=False))
        sys.exit(1)

    ruta = sys.argv[1]

    try:
        # Cargamos expected.json si existe
        expected_json = cargar_esperados_de_json(ruta)

        # Caso 1: ruta es un PDF -> flujo normal
        if os.path.isfile(ruta) and ruta.lower().endswith(".pdf"):
            t0 = time.perf_counter()
            ocr_text = ocr_pdf_to_text(ruta)
            if len(ocr_text) == 0:
                print(json.dumps({"error":"OCR vacío"}, ensure_ascii=False))
                sys.exit(2)

            resultado = infer_from_text(ocr_text)
            t1 = time.perf_counter()
            print(json.dumps(resultado, ensure_ascii=False))

            # Intentamos evaluar si hay esperado
            nombre = os.path.basename(ruta)
            esperado = expected_json.get(nombre) or esperado_por_nombre(nombre)
            if esperado:
                acc = evaluar_y_mostrar(resultado, esperado, titulo=nombre)
                print(f"⏱️ Tiempo {nombre}: {_fmt_secs(t1 - t0)}")
            else:
                print(f"⏱️ Tiempo {nombre} (sin evaluación): {_fmt_secs(t1 - t0)}")


        # Caso 2: ruta es una carpeta -> inferir todos los PDFs
        elif os.path.isdir(ruta):
            pdfs = [os.path.join(ruta, f) for f in os.listdir(ruta) if f.lower().endswith(".pdf")]
            pdfs.sort()
            if not pdfs:
                print(json.dumps({"error":"La carpeta no contiene PDFs"}, ensure_ascii=False))
                sys.exit(3)

            accs = []
            tiempos = []
            print(f"== Procesando carpeta: {ruta} ({len(pdfs)} PDFs) ==")
            t_total0 = time.perf_counter()

            for pdf_path in pdfs:
                nombre = os.path.basename(pdf_path)
                try:
                    t0 = time.perf_counter()
                    ocr_text = ocr_pdf_to_text(pdf_path)
                    if not ocr_text:
                        print(f"[WARN] OCR vacío en {nombre}, se omite evaluación.")
                        continue
                    resultado = infer_from_text(ocr_text)
                    t1 = time.perf_counter()
                    tiempos.append(t1 - t0)

                    print(f"\n--- {nombre} ---")
                    print(json.dumps(resultado, ensure_ascii=False))

                    esperado = expected_json.get(nombre) or esperado_por_nombre(nombre)
                    if esperado:
                        acc = evaluar_y_mostrar(resultado, esperado, titulo=nombre)
                        accs.append(acc)
                    else:
                        print("ℹ️ Sin esperado: no se evalúa ACC.")

                    print(f"⏱️ Tiempo {nombre}: {_fmt_secs(t1 - t0)}")

                except Exception as e:
                    print(f"[ERROR] {nombre}: {e}")

            t_total1 = time.perf_counter()

            if accs:
                media_acc = sum(accs) / len(accs)
                print("\n==============================")
                print(f"📊 ACC medio sobre {len(accs)} evaluaciones: {media_acc:.2%}")
                print(f"⏱️ Tiempo TOTAL carpeta: {_fmt_secs(t_total1 - t_total0)}")
                if tiempos:
                    print(f"⏱️ Tiempo medio por PDF: {_fmt_secs(sum(tiempos)/len(tiempos))}")
                print("==============================")
            else:
                print("\nℹ️ No se calcularon ACCs (no había 'esperados').")
                print(f"⏱️ Tiempo TOTAL carpeta: {_fmt_secs(t_total1 - t_total0)}")


        else:
            print(json.dumps({"error":"Ruta no válida: especifique un PDF o una carpeta"}, ensure_ascii=False))
            sys.exit(4)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


