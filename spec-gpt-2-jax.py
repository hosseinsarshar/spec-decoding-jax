import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
import datetime
import timing_util

def load_model_and_tokenizer(model_name):
    model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_with_topk(
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
    input_ids = tokenizer.encode(prompt, return_tensors="jax")
    print(jax.device_get(input_ids))
    
    # Create a function for generation
    def generate_step(input_ids, key):
        @jax.jit
        def sample_topk(logits):
            if top_k > 0:
                top_k_logits, _ = jax.lax.top_k(logits, k=top_k)
                logits = jnp.where(logits < top_k_logits[:, -1:], -1e10, logits)
            return jax.random.categorical(key, logits / temperature, axis=-1)
        
        outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            prng_key=key,
            use_cache=True
        )
        return outputs.sequences
    
    # Generate text
    key = jax.random.PRNGKey(seed)
    generated = generate_step(input_ids, key)
    
    # Decode the generated ids
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    return generated_texts

model_name = "gpt2"  # You can change this to any other model supported by Hugging Face
model, tokenizer = load_model_and_tokenizer(model_name)

prompt = "Once upon a time"
s = datetime.datetime.now()
generated_texts = generate_with_topk(
    model,
    tokenizer,
    prompt,
    max_length=200,
    top_k=50,
    temperature=0.7,
    num_return_sequences=1
)
e = datetime.datetime.now()
print(f'Compile it in: [{1000*(e-s).total_seconds()}] ms')

s = datetime.datetime.now()
generated_texts = generate_with_topk(
    model,
    tokenizer,
    prompt,
    max_length=200,
    top_k=50,
    temperature=0.7,
    num_return_sequences=1
)
e = datetime.datetime.now()
print(f'The time to execute [{1000*(e-s).total_seconds()}] ms')

max_length = 200
top_k = 50
temperature = 0.7
num_return_sequences = 1

# Pass everything as positional arguments (no kwargs)
args = (model, tokenizer, prompt, max_length, top_k, temperature, num_return_sequences)

timing_util.simple_timeit(generate_with_topk, *args, task='jax-decode')
