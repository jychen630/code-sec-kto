import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
from datetime import datetime

# Paths and settings
model_path = "/local/nlp/junyao/huggingface/20241110_072946_codellama7b_/LATEST/policy.pt"
base_model_name = "codellama/CodeLlama-7b-hf"

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

# First load the state dict
print("Loading state dict...")
state_dict = torch.load(model_path, map_location='cpu')

# Load the base model without quantization first
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_name)
print("Loading state dict into model...")
model.load_state_dict(state_dict['state'])

# Now quantize the loaded model
print("Quantizing model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# Load base model for comparison with same quantization
print("Loading base model for comparison...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

print("Setting models to evaluation mode...")
model.eval()
base_model.eval()
print("Models ready for inference.")

def generate_response(prompt, model, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # Get the device of the first parameter of the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
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

def save_response(response_dir, filename, content):
    """Helper function to save responses to files"""
    os.makedirs(response_dir, exist_ok=True)
    with open(os.path.join(response_dir, filename), 'w') as f:
        json.dump(content, f, indent=2)

def compare_responses(prompt, response_dir="responses/codellama"):
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create response object
    response_obj = {
        "timestamp": timestamp,
        "prompt": prompt,
        "responses": {}
    }
    
    print("\n" + "="*80)
    print("Prompt:")
    print(prompt)
    
    print("\nFine-tuned Model:")
    ft_response = generate_response(prompt, model)
    response_obj["responses"]["fine_tuned"] = ft_response
    print(ft_response)
    
    print("\nBase Model:")
    base_response = generate_response(prompt, base_model)
    response_obj["responses"]["base"] = base_response
    print(base_response)
    print("="*80)
    
    # Save individual response
    save_response(
        response_dir, 
        f"response_{timestamp}.json",
        response_obj
    )
    
    # Update cumulative responses file
    cumulative_file = os.path.join(response_dir, "all_responses.json")
    if os.path.exists(cumulative_file):
        with open(cumulative_file, 'r') as f:
            all_responses = json.load(f)
    else:
        all_responses = []
    
    all_responses.append(response_obj)
    save_response(response_dir, "all_responses.json", all_responses)
    
    print(f"\nResponses saved to {response_dir}/response_{timestamp}.json")
    print(f"Cumulative responses updated in {cumulative_file}")

# Test examples
test_prompts = [
    """int calculateSum(int numbers[], int size) {
    // Add your code here
    
    }""",
    
    """Write a secure password validation function in c/c++""",
    
    """// Create a function to check if a string is a valid email
bool validateEmail(const char* email) {
    
}""",
]

# Create response directory with model name and timestamp
base_dir = f"responses/codellama_{datetime.now().strftime('%Y%m%d')}"

print("\nStarting with test examples...")
for prompt in test_prompts:
    compare_responses(prompt, base_dir)

print("\nEntering interactive mode...")
while True:
    user_input = input("\nEnter your prompt (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    
    compare_responses(user_input, base_dir)

print("\nTesting completed!")