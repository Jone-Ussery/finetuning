"""
Example script to download any arbitrary model and format the repo correctly.
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import dotenv


def save_model(model, tokenizer, path: str, model_name: str):
    """Saves a model and tokenizer to a path.

    Args:
        model: The model to save.
        tokenizer: The tokenizer to save.
        path (str): The path to save the model and tokenizer to.
    """
    # make sure the path exists
    os.makedirs(path, exist_ok=True)

    model_path = os.path.join(path, model_name)
    model.save_pretrained(model_path, max_shard_size='2GB')
    tokenizer.save_pretrained(model_path)
    print(f"Model and tokenizer saved to {path}")
    
dotenv.load_dotenv()

login(
    token=os.environ["HF_ACCESS_TOKEN"],
)


model_name = 'mistralai/Mistral-7B-v0.1'
save_path = 'bittensor_models'
model_dir_name = model_name.split('/')[1]

print(f"Loading model {model_name}")
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer
print(f"Saving model {model_name}")
save_model(model, tokenizer, save_path, model_dir_name)
print(f"Model {model_name} saved to {save_path}")
