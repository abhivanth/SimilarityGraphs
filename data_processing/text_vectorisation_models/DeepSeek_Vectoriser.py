import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_deepseek_model(model_type, model_variant=None):
    # Define the available models and their variants
    model_map = {
        "qwen": {
            "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        },
        "llama": {
            "8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8Bb",
            "70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        }
    }

    if model_type.lower() not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Get the model variant if specified, otherwise use the first variant
    if model_variant:
        model_name = model_map[model_type.lower()].get(model_variant.lower())
        if not model_name:
            raise ValueError(f"Unsupported model variant: {model_variant}")
    else:
        model_name = list(model_map[model_type.lower()].values())[0]  # Default to the first variant if none specified

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    return tokenizer, model


def mean_pooling(output, attention_mask):
    logits = output.logits
    token_embeddings = logits
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1),
                                                                                  min=1e-9)


def embed_text(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        print("Inputs moved to CUDA (GPU)")
    else:
        print("CUDA not available. Inputs remain on CPU.")

    with torch.no_grad():
        output = model(**inputs)
    return mean_pooling(output, inputs["attention_mask"]).cpu().numpy()


def vectorize_dataframe(df, text_column, id_column, model_type, model_variant=None):
    tokenizer, model = load_deepseek_model(model_type, model_variant)
    vectors = []
    for _, row in df.iterrows():
        embedding = embed_text(row[text_column], tokenizer, model)
        text_id = np.array([row[id_column]], dtype=np.float32)
        combined = np.column_stack((embedding, text_id))
        vectors.append(combined)
    return np.vstack(vectors)
