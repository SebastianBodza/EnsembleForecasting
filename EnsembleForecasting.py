import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from IPython.display import clear_output, display, HTML

model_name_or_path = "TheBloke/Magicoder-S-DS-6.7B-AWQ"
tokenizer1 = AutoTokenizer.from_pretrained(model_name_or_path)
model1 = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)

model_name_or_path = "TheBloke/deepseek-coder-6.7B-instruct-AWQ"
tokenizer2 = AutoTokenizer.from_pretrained(model_name_or_path)
model2 = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)

prompt_template1="""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

prompt_template2="""You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction}
### Response:
{response}"""



def generate_completions(n, prompt):
  """Comment out the comments to get a color styled output in jupyter notebooks"""
    input_text = prompt
    response = ""
    display_text = ""
    generated_tokens_list = []

    for _ in range(n):
        # clear_output(wait=True)
        
        input_text1 = prompt_template1.format(instruction=input_text, response=response)
        input_text2 = prompt_template2.format(instruction=input_text, response=response)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(run_model_prediction, tokenizer1, model1, input_text1)
            future2 = executor.submit(run_model_prediction, tokenizer2, model2, input_text2)
            
            generated_tokens1, probability1, t_prob1 = future1.result()
            generated_tokens2, probability2, t_prob2 = future2.result()
        
        if probability1 > probability2:
            gen_token = tokenizer1.decode(generated_tokens1[0])
            color = '#93f5af'  # green
        else:
            gen_token = tokenizer2.decode(generated_tokens2[0])
            color = '#93c4f5'  # blue

        if gen_token == "<|EOT|>":
            break
        if gen_token == " <｜end▁of▁sentence｜>":
            break
        response += gen_token
        html_token = gen_token.replace("\n", "<br>")
        display_text += f'<span style="color: {color}">{html_token}</span>'
        # display(HTML(display_text.replace(r"\n", "<br>")))
        
        generated_tokens_list.append(gen_token)
        
        # print(f"| {tokenizer1.decode(generated_tokens1[0]):8s} | {probability1:.4f} | {probability1:.2%}")
        # print(f"| {tokenizer2.decode(generated_tokens2[0]):8s} | {probability2:.4f} | {probability2:.2%}")
        if response.endswith("\n\n\n"):
            break
          
    print(response)
    return response


n = 10
completions = generate_completions(n, "Create a react exxample with Tailwind")
print(completions)

# Generating humanEval
# TODO: FIX PROPER PROMPTING FOR HUMANEVAL!

from human_eval.data import write_jsonl, read_problems

problems = read_problems()
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm 

def generate_one_completion(prompt: str):
  return generate_completions(300, prompt)

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]
write_jsonl("combined.jsonl", samples)

