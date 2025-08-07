from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer
)
import torch

# Cargar dataset desde JSONL
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# Usamos t5-base en vez de t5-small, que tiene un índice mas elevado de ACC.
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocesamiento
def preprocess(example):
    input_text = example["input"]
    target_text = example["output"]
    model_inputs = tokenizer(
        input_text, max_length=1024, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        target_text, max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Aplicamos el preprocesamiento
dataset = dataset.map(preprocess, batched=False)

# Argumentos de entrenamiento optimizados para tarjeta gráfica nivél media/alta
training_args = TrainingArguments(
    output_dir="./t5_factura_base_model",
    per_device_train_batch_size=2,           # Reduce el uso de VRAM
    gradient_accumulation_steps=2,           # Simula batch_size=4
    num_train_epochs=4,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    fp16=True,                               # Usa half precision en GPU
    report_to="none"                         # Evita errores si no usas wandb
)

# Configurar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Entrenar
trainer.train()

# Guardar modelo y tokenizer
model.save_pretrained("./t5_factura_base_model")
tokenizer.save_pretrained("./t5_factura_base_model")
