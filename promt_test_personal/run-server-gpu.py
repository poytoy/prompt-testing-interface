import gradio as gr
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import json
import gc
# Function to load a model
#created by poyraz guler
class ModelHandler:
    def __init__(self,max_length=4096):
        self.model_cache = {}
        self.tokenizer_cache = {}
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_path = None
        self.current_model=None
        self.current_tokenizer = None

    def unload_model(self):
        """Unload all model from the GPU."""
        if self.current_model is not None:
            if "4bit" not in self.current_model_path and "8bit" not in self.current_model_path:
               self.current_model.to('cpu')
            del self.current_model
            del self.current_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_path = None
    def load_model(self, model_path,hf_token):
        # Check if the requested model is the current one
        if model_path == self.current_model_path:
            # Return the already loaded model and tokenizer
            return self.current_model, self.current_tokenizer
        # Unload the current model if it's different from the requested model
        
        self.unload_model()
        """Load and cache the model and tokenizer."""
        
        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True)
        #for quantized model
        if "4bit" in model_path:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
        #for nonquantized model
            model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)                                                             

        self.current_model_path = model_path
        self.current_model = model
        self.current_tokenizer = tokenizer

        return model, tokenizer
    def format_prompt(self, document_text, prompt, model_path):
        """Format the prompt based on the model path."""
        if model_path == "core-outline/gemma-2b-instruct":
            return f"Instruction: {prompt}\n\nContext: {document_text}"
        elif model_path == "unsloth/Yi-1.5-6B-bnb-4bit":
            return f"{document_text} {prompt} "
        elif model_path == "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit":
            return f"system\n\n{document_text}\n\n{prompt}\n\n"
        
    def infer(self, model, tokenizer, input_text,generation_config,max_new_tokens, temperature):
        """Perform inference using the model and tokenizer."""
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            padding=True, 
            max_length=self.max_length,
            pad_to_multiple_of=self.max_length
        ).to(self.device)
        #accomidate for different tokeniztions of different llm

        generation_params = {
        "max_new_tokens": max_new_tokens,
        "do_sample": generation_config.get("do_sample", True),
        "temperature": temperature,
        "top_p": generation_config.get("top_p", 0.9),
        "top_k": generation_config.get("top_k", 50),
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_params
            )

        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Post-process the decoded outputs
        processed_outputs = "\n".join([line.strip() for line in decoded_outputs])
        
        word_count = len(processed_outputs.split())
        char_count = len(processed_outputs)
        token_count = len(tokenizer.encode(processed_outputs, add_special_tokens=False))

        # Check if generation stopped because of eos_token or max tokens
        stopped_by_eos = any(tokenizer.eos_token_id in output for output in outputs.tolist())

        metrics = {
            "word_count": word_count,
            "char_count": char_count,
            "token_count": token_count,
            "stopped_by_eos": stopped_by_eos
        }

        return processed_outputs, metrics



# Load the configuration file
class ConfigLoader:
    @staticmethod
    def load_config():
        """Load the configuration from a JSON file."""
        config_path = 'config.json'
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

# Gradio interface function
def process_input(model_path, document_file,prompt1,prompt2,max_new_tokens,temperature,model_handler,generation_configs,hf_token):
    loader = TextFileLoader(document_file.name)
    docs = loader.load_and_split()
    """Process input by loading the model and performing inference."""
    model, tokenizer = model_handler.load_model(model_path,hf_token)
    
    document_text = " ".join([doc['page_content'] for doc in docs])

    
    prompt1_filled = model_handler.format_prompt(document_text, prompt1, model_path)
    prompt2_filled = model_handler.format_prompt(document_text, prompt2, model_path)



    generation_config = generation_configs.get(model_path, {})
    generation_config["max_new_tokens"] = max_new_tokens
    generation_config["temperature"] = temperature

    output1, metrics_1 = model_handler.infer(model, tokenizer, prompt1_filled,generation_config, max_new_tokens, temperature)
    output2, metrics_2 = model_handler.infer(model, tokenizer, prompt2_filled,generation_config, max_new_tokens, temperature)
    return(output1, metrics_1["word_count"],metrics_1["char_count"],metrics_1["token_count"],metrics_1["stopped_by_eos"],
           output2,metrics_2["word_count"],metrics_2["char_count"],metrics_2["token_count"],metrics_2["stopped_by_eos"])

class TextFileLoader:
    def __init__(self, file_path, chunk_size=1000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def load_and_split(self):
        """Load and split the text file into document chunks."""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Check if the document text length exceeds the chunk size
        if len(text) <= self.chunk_size:
            # Return the entire document as a single chunk if it's small enough
            return [{'page_content': text}]
        
        # Split the text into chunks of `chunk_size` length
        chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return [{'page_content': chunk} for chunk in chunks]

# Main function to set up and launch the Gradio interface
def main():
    # Load configuration
    config = ConfigLoader.load_config()
    model_paths = config.get("models", [])
    generation_configs = config.get("generation_configs", {})
    hf_token = config.get("hf_token", None)

    model_handler = ModelHandler(max_length=512)

    def get_defaults(model_path):
        defaults = generation_configs.get(model_path, {})
        max_new_tokens = defaults.get("max_new_tokens", 2048)
        temperature = defaults.get("temperature", 1.0)
        return max_new_tokens, temperature

    def update_defaults(model_path):
        max_new_tokens, temperature = get_defaults(model_path)
        
        # Create updates for each component
        return gr.Number.update(value=max_new_tokens), gr.Number.update(value=temperature)


    # Define the interface
    with gr.Blocks() as demo:
            model_dropdown = gr.Dropdown(choices=model_paths, label="Select Model")
            document_file_input = gr.File(label="Select txt file")
            prompt_text1_input = gr.Textbox(label="Prompt 1")
            prompt_text2_input = gr.Textbox(label="Prompt 2")
            max_new_tokens_input = gr.Number(label="Max New Tokens")
            temperature_input = gr.Number(label="Temperature")
            output_box1 = gr.Textbox(label="Output 1")
            output_box2 = gr.Textbox(label="Output 2")
            word_count1 = gr.Number(label="Word Count 1")
            char_count1 = gr.Number(label="Character Count 1")
            token_count1 = gr.Number(label="Token Count 1")
            stopped_by_eos1 = gr.Checkbox(label="Stopped by EOS 1")
            word_count2 = gr.Number(label="Word Count 2")
            char_count2 = gr.Number(label="Character Count 2")
            token_count2 = gr.Number(label="Token Count 2")
            stopped_by_eos2 = gr.Checkbox(label="Stopped by EOS 2")
            
            def update_defaults(model_path):
                max_new_tokens, temperature = get_defaults(model_path)
                return gr.update(value=max_new_tokens), gr.update(value=temperature)

            # Bind the update function to the model dropdown change event
            model_dropdown.change(
                lambda model_path: update_defaults(model_path),
                inputs=model_dropdown,
                outputs=[max_new_tokens_input, temperature_input]
            )
            # Set up the main function
            gr.Interface(
                fn=lambda model_path, document_file, prompt1, prompt2, max_new_tokens, temperature: process_input(model_path, document_file, prompt1, prompt2, max_new_tokens,temperature,model_handler,generation_configs,hf_token),
                inputs=[model_dropdown, document_file_input, prompt_text1_input, prompt_text2_input, max_new_tokens_input, temperature_input],
                outputs=[output_box1, word_count1, char_count1, token_count1, stopped_by_eos1, output_box2, word_count2, char_count2, token_count2, stopped_by_eos2],
                live=False,
                clear_btn=None
            )

        # Launch the interface
    demo.launch(share=True)

if __name__ == "__main__":
    main()
