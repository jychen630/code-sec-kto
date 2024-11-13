import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import datetime
# Paths and settings
model_path = "/scratch/jc9723/huggingface/20241111_044108_starcoder_/LATEST/policy.pt"
base_model_name = "bigcode/starcoderbase-1b"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,4,7"
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
model.load_state_dict(state_dict['state'])
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

def save_response(response_dir, filename, content):
    """Helper function to save responses to files"""
    os.makedirs(response_dir, exist_ok=True)
    with open(os.path.join(response_dir, filename), 'w') as f:
        json.dump(content, f, indent=2)

def compare_responses(prompt, response_dir="responses/starcoder"):
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
    
    """Write a secure password validation function""",
    
    """// Create a function to check if a string is a valid email
bool validateEmail(const char* email) {
    
}""",
]
# Interactive testing loop
print("\nStarting with test examples...")
for prompt in test_prompts:
    compare_responses(prompt)



print("\nTesting completed!")