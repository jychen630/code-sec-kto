import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths and settings
model_path = "/local/nlp/junyao/huggingface/20241110_014700_codellama7b_/LATEST/policy.pt"
base_model_name = "codellama/CodeLlama-7b-hf"

# Load tokenizer and models
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

# Load fine-tuned model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_name)

state_dict = torch.load(model_path)
print("Fine-tuned model state_dict loaded.")
print("Attaching fine-tuned state_dict to base model...")
model.load_state_dict(state_dict)
print("Attached fine-tuned state_dict to base model.")
print("Setting fine-tuned model to evaluation mode...")
model.eval()
print("Fine-tuned model set to evaluation mode.")
print("Moving fine-tuned model to GPU...")
model.cuda()
print("Fine-tuned model moved to GPU.")
print("="*80)

# Load base model for comparison
print("Loading base model for comparison...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).cuda()
print("Base model loaded (for comparison).")

def generate_response(prompt, model, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def compare_responses(prompt):
    print("\n" + "="*80)
    print("Prompt:")
    print(prompt)
    
    print("\nFine-tuned Model:")
    ft_response = generate_response(prompt, model)
    print(ft_response)
    
    print("\nBase Model:")
    base_response = generate_response(prompt, base_model)
    print(base_response)
    print("="*80)

# Test examples
test_prompts = [
    """def calculate_sum(numbers):
    # Add your code here
    """,
    
    """Write a secure password validation function""",
    
    """# Create a function to check if a string is a valid email
def validate_email(email):
    """,
]

# Interactive testing loop
print("\nStarting with test examples...")
for prompt in test_prompts:
    compare_responses(prompt)

print("\nEntering interactive mode...")
while True:
    user_input = input("\nEnter your prompt (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    
    compare_responses(user_input)

print("\nTesting completed!")