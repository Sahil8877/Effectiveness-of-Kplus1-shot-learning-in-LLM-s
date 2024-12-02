def llama_call(llm, prompt, temperature):
    """
    Calls the Llama model to generate a response based on the input prompt.
    """
    output = llm(
        prompt,  # Prompt string passed to the model
        max_tokens=200,  # Generate up to 200 tokens
        stop=["STOP"],  # Stop generating before new questions
        echo=False,  # Don't echo the prompt back in the output
        logprobs=50,  # Get the log probabilities for the top tokens
        top_k=50,  # Set the top_k sampling method for generation
        temperature=temperature,  # Sampling temperature
    )  # Generate completion

    return output


def load_llama():
    """
    Load the Llama model and prepare it for inference.
    """
    from llama_cpp import Llama
    import torch
    
    print(torch.__version__)  # Print torch version for debugging
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Automatically use GPU if available
    print(device)  # Print selected device (CUDA or CPU)
    
    llm = Llama(
        model_path="llama_tools/Meta-Llama-3-8B-Instruct.Q8_0.gguf",  # Model file path
        logits_all=True,  # Whether to return logits for all tokens
        verbose=False,
        n_gpu_layers=-1,  # Set to -1 to use all available GPU layers
        n_ctx=2048,  # Context window size (can be increased if needed)
    )

    llm.set_seed(1000)  # Set the random seed for reproducibility
    return llm


def single_call(llm, prompt, temperature):
    """
    Generate a response from the Llama model based on the prompt.
    """
    # Call the Llama model
    output = llama_call(llm, prompt, temperature)

    # Log probabilities and token probabilities
    token_logprobs = output['choices'][0]['logprobs']['token_logprobs']
    prob_seq = sum(token_logprobs)  # Summing log probabilities of tokens

    # Extract the generated text (the model's response)
    answer = output['choices'][0]['text']

    # Return both the answer and the probability sequence
    result = {"answer": answer, "prob_seq": float(prob_seq)}
    return result


