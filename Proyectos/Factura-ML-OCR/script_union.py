import json

# Creamos el archivo final
with open("dataset_final.jsonl", "w", encoding="utf-8") as outfile:
    for i in range(1, 20):  # del 1 al 19
        filename = f"dataset{i}.jsonl"
        print(f"Procesando {filename}...")
        try:
            with open(filename, "r", encoding="utf-8") as infile:
                for line in infile:
                    # Simplemente copiamos línea a línea
                    outfile.write(line)
        except FileNotFoundError:
            print(f"⚠ El archivo {filename} no se ha encontrado, se omite.")
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")

print("✅ Dataset final generado: dataset_final.jsonl")
