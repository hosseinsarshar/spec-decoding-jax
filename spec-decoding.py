import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
import numpy as np
import os

# Load both models: the draft (smaller) and the verifier (larger)
draft_model_name = "meta-llama/Meta-Llama-3-8B"  # e.g., "facebook/opt-1.3b"
verifier_model_name = "meta-llama/Meta-Llama-3-8B"  # e.g., "facebook/opt-13b"
hf_token = os.getenv("HF_TOKEN")
print(f'{hf_token=}')
draft_model = FlaxAutoModelForCausalLM.from_pretrained(draft_model_name, use_auth_token=hf_token)
verifier_model = FlaxAutoModelForCausalLM.from_pretrained(verifier_model_name, use_auth_token=hf_token)

tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

# Function to generate tokens with the draft model
def draft_generate(draft_model, input_ids, num_candidates=5):
    # Get logits for the next token (generate num_candidates tokens)
    outputs = draft_model(input_ids)
    logits = outputs.logits[:, -1, :]  # Get logits for the last token
    
    # Sample num_candidates tokens based on logits
    candidate_tokens = jax.lax.top_k(logits, num_candidates).indices
    return candidate_tokens

# Function to verify tokens with the verifier model
def verify_tokens(verifier_model, input_ids, candidate_tokens):
    logits_list = []
    
    def get_logits_for_token(token_idx):
        candidate_input_ids = jnp.concatenate((input_ids, candidate_tokens[:, token_idx].reshape(-1, 1)), axis=1)
        outputs = verifier_model(candidate_input_ids)
        logits = outputs.logits[:, -1, :]
        return logits
    
    # Calculate logits for each candidate token
    for i in range(candidate_tokens.shape[1]):
        logits = get_logits_for_token(i)
        logits_list.append(logits)
    
    # Stack logits for all candidate tokens
    logits_stack = jnp.stack(logits_list, axis=1)
    
    # For each candidate token, get the logit corresponding to its position
    candidate_scores = jnp.take_along_axis(logits_stack, candidate_tokens[..., None], axis=-1).squeeze(-1)
    
    # Select the highest-scoring token
    best_token = jnp.argmax(candidate_scores, axis=-1)
    return candidate_tokens[jnp.arange(candidate_tokens.shape[0]), best_token]

# Main speculative decoding loop
def speculative_decode(draft_model, verifier_model, input_text, max_length=50, num_candidates=5):
    input_ids = tokenizer(input_text, return_tensors='jax').input_ids
    
    for _ in range(max_length):
        # Generate candidate tokens with the draft model
        candidate_tokens = draft_generate(draft_model, input_ids, num_candidates=num_candidates)
        
        # Verify and select the best token with the verifier model
        best_token = verify_tokens(verifier_model, input_ids, candidate_tokens)
        
        # Append the best token to the input
        input_ids = jnp.concatenate([input_ids, best_token.reshape(-1, 1)], axis=-1)
        
        # Stop if the end-of-sentence token is generated
        if best_token == tokenizer.eos_token_id:
            break
    
    # Decode and return the final output text
    output_text = tokenizer.decode(np.array(input_ids[0]), skip_special_tokens=True)
    return output_text

# Example usage
input_text = "Once upon a time"
output = speculative_decode(draft_model, verifier_model, input_text, max_length=50, num_candidates=5)
print(output)
