from model import NGCTransformer
import jax
import jax.numpy as jnp
import numpy as np
from config import Config as config
from data_preprocess.data_loader import DataLoader
import tiktoken


# Initialize the model
dkey = jax.random.PRNGKey(0)
model = NGCTransformer(dkey, batch_size=config.batch_size, seq_len=config.seq_len, n_embed=config.n_embed, vocab_size=config.vocab_size, n_layers=config.n_layers, n_heads=config.n_heads,
                          T=config.num_iter, dt=1., tau_m=config.tau_m , act_fx=config.act_fx, eta=config.eta, dropout_rate= config.dropout_rate, exp_dir="exp",
                  loadDir= None, pos_learnable= config.pos_learnable, optim_type=config.optim_type, wub =config.wub, wlb= config.wlb, model_name="ngc transformer" )


# Initialize the encoder
enc = tiktoken.get_encoding(config.tokenizer_name)  # Use the tokenizer name from config

def generate_text(
    model,
    prompt: str,
    max_new_tokens: int = 100,
    seq_len: int = 8,
    temperature: float = 1.0,
    key=None
):
    # Encode prompt
    prompt_ids = enc.encode_ordinary(prompt)
    prompt_tensor = jnp.array([prompt_ids], dtype=jnp.int32)  # shape: (1, seq_len)

    current_tokens = prompt_tensor
    current_key = key

    for _ in range(max_new_tokens):
        # Truncate or pad to fit within seq_len
        if current_tokens.shape[1] > seq_len:
            input_seq = current_tokens[:, -seq_len:]
        else:
            input_seq = current_tokens

        # Pad to exactly seq_len if needed 
        if input_seq.shape[1] < seq_len:
            pad_len = seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=0)
        dummy_target = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size))  

        # Run inference 
        y_mu_inf, _, _ = model.process(input_seq, dummy_target, adapt_synapses=False)

        logits = y_mu_inf.reshape(config.batch_size, config.seq_len, config.vocab_size)

        # Get logits for the last **real** token in the input (not padding)
        actual_len = min(current_tokens.shape[1], seq_len)
        last_pos = actual_len - 1
        next_logits = logits[0, last_pos, :] / temperature

        if current_key is not None:
            probs = jax.nn.softmax(next_logits)
            current_key, subkey = jax.random.split(current_key)
            next_token = jax.random.choice(subkey, a=config.vocab_size, p=probs)
        else:
            next_token = jnp.argmax(next_logits)

        # Append token
        current_tokens = jnp.concatenate([current_tokens, next_token[None, None]], axis=1)

    # Decode full sequence to string
    generated_ids = current_tokens[0].tolist()
    return enc.decode(generated_ids)

# Example usage
prompt = "The king said: "
generated = generate_text(
    model=model,
    prompt=prompt,
    max_new_tokens=200,
    seq_len=config.seq_len,        
    temperature=0.8,
    key=jax.random.PRNGKey(42)  
)
print(generated)