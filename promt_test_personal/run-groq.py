
import os
import textwrap
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
import inspect

def create_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set the GROQ_API_KEY environment variable")
    return Groq(api_key=api_key)

def clean_doc(doc):
    # Ensure doc is a string before applying expandtabs
    if not isinstance(doc, str):
        raise TypeError("Expected a string for doc, but got a different type.")
    
    # Clean the doc by expanding tabs and splitting by newlines
    lines = doc.expandtabs().split('\n')
    return lines
# Initialize Groq client


def generate_text(client, prompt, file_content, model_name):
    combined_prompt = f"{prompt}\n\n{file_content}"
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": combined_prompt,
                }
            ],
            model=model_name,
        )
        response_text = chat_completion.choices[0].message.content
        return response_text
    except Exception as e:
        print(f"Error generating text with Groq: {e}")
        return "Error generating text."

class TextFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self, chunk_size=1000):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split text into chunks
        docs = [{'page_content': page} for page in text.split('\n\n')]
        return docs
    
def ExtractQuestions_txt(client,system_input, prompt_template,model_name):
    try: 
        loader = TextFileLoader(system_input.name)
        docs = loader.load_and_split()
    
        document_text = " ".join([doc['page_content'] for doc in docs])
    except Exception as e:
        document_text = system_input

    # Define the prompt for question extraction and analysis
    prompt = prompt_template.format(document_text=document_text)

    # Generate text using the model
    try:
        response_text = generate_text(client,prompt, document_text,model_name)
        
        if isinstance(response_text, dict):
            response_text = response_text.get('text', '')

        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Ensure proper Markdown formatting if needed
        processed_text = inspect.cleandoc(response_text)
        
        return processed_text
    except Exception as e:
        print(f"Error processing the prompt: {e}")
        return "Error processing the prompt."
          

def main():
    #client initilization with function
    load_dotenv()
    client = create_client()
    #gradio input outputs
    input_file = gr.File(label="Select txt file")
    input_txt = gr.Textbox(label="input text",placeholder="write text instead")
    prompt_1 = gr.Textbox(label="Prompt Simple", placeholder="Enter your first prompt here")
    prompt_2 = gr.Textbox(label="Prompt Complicated", placeholder="Enter your second prompt here")
    output_1 = gr.Textbox(label="Output (Prompt 1)")
    output_2 = gr.Textbox(label="Output (Prompt 2)")
    if input_file is not None:
        system_input=input_file
    else:
        system_input=input_txt

    models = ["llama3-70b-8192","llama3-8b-8192","mixtral-8x7b-32768","gemma-7b-it","gemma2-9b-it"]
    loaded_model=gr.State()
    def load_model(model_name):
        loaded_model.value=model_name
        return f"{model_name} loaded succesfully"
    def compare_prompts(system_input, prompt_1, prompt_2):
        if loaded_model.value == None:
            return "Model not loaded. Please load the model first.", ""
        result_1 = ExtractQuestions_txt(client,system_input, prompt_1,loaded_model.value)
        result_2 = ExtractQuestions_txt(client,system_input, prompt_2,loaded_model.value)
        return result_1,result_2
        
    with gr.Blocks() as interface:
        load_button = gr.Button("Load Model")
        selected_model=gr.Dropdown(choices=models,label="select1 model")
        load_message= gr.Textbox(label="model load status")

        load_button.click(load_model, inputs=[selected_model], outputs=[load_message])

        with gr.Tabs():
            with gr.TabItem("Upload File"):
                gr.Markdown("### Upload a text file and provide two prompts")
                file_interface = gr.Interface(
                    fn=compare_prompts,
                    inputs=[
                        gr.File(label="Select txt file"),
                        gr.Textbox(label="Prompt Simple", placeholder="Enter your first prompt here"),
                        gr.Textbox(label="Prompt Complicated", placeholder="Enter your second prompt here"),
                    ],
                    outputs=[
                        gr.Textbox(label="Output (Prompt 1)"),
                        gr.Textbox(label="Output (Prompt 2)")
                    ]
                )
                
            with gr.TabItem("Enter Text"):
                gr.Markdown("### Enter text directly and provide two prompts")
                text_interface = gr.Interface(
                    fn=compare_prompts,
                    inputs=[
                        gr.Textbox(label="Input Text", placeholder="Write text instead"),
                        gr.Textbox(label="Prompt Simple", placeholder="Enter your first prompt here"),
                        gr.Textbox(label="Prompt Complicated", placeholder="Enter your second prompt here"),
                    ],
                    outputs=[
                        gr.Textbox(label="Output (Prompt 1)"),
                        gr.Textbox(label="Output (Prompt 2)")
                    ]
                )

    
    interface.launch(share=True)

if __name__ == "__main__":
    main()