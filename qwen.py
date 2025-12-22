import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Модель Qwen 2.5 не потребує дозволів і працює "з коробки"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# Завантаження моделі та токенізатора
print("--- Loading model (this may take a few minutes the first time) ---")
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto", # Автоматично обере bfloat16 або float32
    device_map="auto",   # Використає GPU, якщо він є
)

# Промпт для вашого додатка
messages = [
    {
        "role": "system",
        "content": (
            "You are a specialized Botanical AI Assistant. "
            "Provide fascinating, scientifically accurate, and engaging facts. "
            "If specific species details are sparse, provide information about the genus or family."
        )
    },
    {
        "role": "user",
        "content": "Tell me something interesting about Leptinella potentillina."
    },
]

print("--- Generating Fact ---")
outputs = pipe(
    messages,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
)

# Виведення результату
print("\nБОТАНІЧНА ДОВІДКА:")
print(outputs[0]["generated_text"][-1]["content"])