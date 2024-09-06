
# Welcome
Welcome to Prompt Engineering Test for Low Parameter LLMs. created by Poyraz Güler.

Here we have to main codes separate from each other. For users who have access to powerful hardware that could run 4bit 8Billion param models we recommend run-server-gpu.py
For users who don't wish to run the models locally we recommend run-api.py which uses GROQ api to run llama8B. Notice that run-api only tests prompts on 1 model and does not have a way to add your own model.



# SERVER
Test out llama3-8b,gemma-2b and Yi-6b models with two different turkish prompts one simple one more complicated. The aim of the test is to see if prompt engineering principles are applicable in turkish prompts for low parameter models. Using Gradio for UI was our choice for better testing. 
You can use this program to add more models and use your own prompts to test for yourself if your models response can be improved with complicated prompts

keep in mind that models here are either ran locally or with a remote server. We did not use api's. Keep in mind that your local hardware must have the models required ram and memory to run this code smoothly.

The code has manual methods to move models from cpu to gpu. we use cuda since our machine has an NVIDIA GPU. don't forget to tailor the code to your specific hardware. 

1. Install the required dependencies:

   """bash
   pip install -r requirements_server.txt
   """
2. Set your Huggingface token in config.json file.
   """
   "hf_token": <"your-huggingface-token">
   """
3. Run your code either locally or in a remote machine.
   """bash
   python run-server-gpu.py
   """
4. Open the link provided in terminal
   4.1. if run locally usually:http://127.0.0.1:7860
   4.2. if run on remote server check the public URL that looks something like:https://8114e06e6176fa0c48.gradio.live
5. Pick a model through the gradio interface and upload a .txt file you want as a system input.


## Results from our test run

All models have been tested for both simple and complicated prompts for 3 scenarios. We test to see if complicated prompts produce better results.

1. system input one: A short made up text. task: extract questions.
Gemma2B: produced slightly better results.
llama8B: no improvement.
Yi6B:    no improvement.

2. system input two: long real text. task: extract information.
Gemma2B: no improvement.
llama8B: no improvement.
Yi6B:    no improvement.

3. system input three: short real text. task: extract information.
Gemma2B: no improvement.
llama8B: produced better result.
Yi6B:    no improvement.

4. system input four: mid sized real text. task: extract information.
Gemma2B: no improvement.
llama8B: no improvement.
Yi6B:    no improvement.

5. system input five: name of a book. task: extract information.
Gemma2B: produced slighltly better results.
llama8B: no improvement.
Yi6B:    no improvement.

After these tests consistent improvement on results is not observed. we conclude that as it stands now our models do not produce better results with more complicated prompts. until further testing done by turkish prompts crafted carefully by turkish prompt engineers we do not recommend the usage of complicated prompts on low parameter large language models.


## Further improvements
* You can add your own model from huggingface by adding the model name to "models":[..] in config.json keep in mind that code might need alterations for different models. you would have to specify the prompt format if you wish you model to not use a generic one. While unloading models from gpu, quantized models are deleted while nonquantized models are moved to cpu. You might need to adjust this process based on your specific model type.
* Our prompts are Deepl translations from a prompt engineering repo https://github.com/danielmiessler/fabric/tree/main . If the user wants to translate the prompt manually it could lead to better results.
* https://console.anthropic.com/dashboard could be used to create new complicated prompts.
* Configurations of models could have an impact on outputs. you could alter the config.json file to test prompts on different settings of the model.












# MAIN
Test out various models provided by Groq api with two different turkish prompts one simple one more complicated. Test to see if prompt engineering logic can be applied to turkish prompts.Using Gradio for UI and Groq api for using llama3-70B and various other models without burdening the hardware
You can use this code to test your own prompts and see for yourself.



1. Install the required dependencies:

   """bash
   pip install -r requirements_api.txt
   """

2. Set the `GROQ_API_KEY` in a .env` file in the same directory as the script:

   """
   GROQ_API_KEY=your-GROQ-api-key
   """

   Run the application:

   """bash
   python run-api.py
   """

3. Open the local link provided in terminal
usually:http://127.0.0.1:7860

4. Upload the selected .txt file or write your own text by selecting the option from user interface and test your prompts.

