import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import timing_util
import datetime

def torch_load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def torch_generate_with_topk(
    model,
    tokenizer,
    prompt,
    max_length=50,
    top_k=50,
    temperature=1.0,
    num_return_sequences=1,
    seed=42
):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Create a function for generation
    def generate_step(input_ids):
        def sample_topk(logits):
            if top_k > 0:
                top_k_logits, _ = torch.topk(logits, k=top_k)
                logits = torch.where(logits < top_k_logits[:, -1:], -1e10, logits)
            return torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
        
        outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            use_cache=True
        )
        return outputs
        
    # Generate text
    torch.manual_seed(seed)
    generated = generate_step(input_ids)
    
    # Decode the generated ids
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    return generated_texts

# Example usage
device = "cuda"
torch_model_name = "gpt2"  # You can change this to any other model supported by Hugging Face
torch_model, torch_tokenizer = torch_load_model_and_tokenizer(torch_model_name)
torch_model.to(device)

generated_texts = torch_generate_with_topk(
    torch_model,
    torch_tokenizer,
    prompt,
    max_length=200,
    top_k=50,
    temperature=0.7,
    num_return_sequences=1
)
e = datetime.datetime.now()

prompt = "Once upon a time"
s = datetime.datetime.now()
generated_texts = torch_generate_with_topk(
    torch_model,
    torch_tokenizer,
    prompt,
    max_length=200,
    top_k=50,
    temperature=0.7,
    num_return_sequences=1
)
e = datetime.datetime.now()
print(f'The time before timeit [{1000*(e-s).total_seconds()}] ms')

average_time_ms = timing_util.simple_timeit(torch_generate_with_topk, torch_model,
    torch_tokenizer,
    prompt,
    max_length=200,
    top_k=50,
    temperature=0.7,
    num_return_sequences=1, task='fsdp_feed_f')

print(f"{average_time_ms=}")