import torch 
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

model_name_or_path = "TheBloke/WizardCoder-33B-V1.1-AWQ"
tokenizer1 = AutoTokenizer.from_pretrained(model_name_or_path)
model1 = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)

model_name_or_path = "TheBloke/deepseek-coder-33B-instruct-AWQ"
tokenizer2 = AutoTokenizer.from_pretrained(model_name_or_path)
model2 = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:1"
)

prompt_template1 = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"

prompt_template2 ="""You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{instruction}
### Response:
{response}"""


class EnsembleModel(nn.Module): 
    def __init__(self, model1, model2, tokenizer1, tokenizer2, prompt_template1, prompt_template2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

        self.prompt_template1 = prompt_template1
        self.prompt_template2 = prompt_template2

    def forward(self, instruction, output):
        text_1 = self.prompt_template1.replace("{instruction}", instruction).replace("{response}", output)
        text_2 = self.prompt_template2.replace("{instruction}", instruction).replace("{response}", output)

        inputs1 = self.tokenizer1(text_1, return_tensors="pt").to(self.model1.device)
        inputs2 = self.tokenizer2(text_2, return_tensors="pt").to(self.model2.device)

        with torch.no_grad():
            outputs1 = self.model1(**inputs1)
            outputs2 = self.model2(**inputs2)

        logits1 = outputs1.logits[:, -1, :]
        logits2 = outputs2.logits[:, -1, :]

        probs1 = torch.nn.functional.softmax(logits1, dim=-1)
        probs2 = torch.nn.functional.softmax(logits2, dim=-1)

        max_prob1, max_indices1 = torch.max(probs1, dim=-1)
        max_prob2, max_indices2 = torch.max(probs2, dim=-1)

        if max_prob1.cpu()>max_prob2.cpu():
            return self.tokenizer1.decode(max_indices1)
        else:
            return self.tokenizer2.decode(max_indices2)

ensemble = EnsembleModel(model1, model2, tokenizer1, tokenizer2, prompt_template1, prompt_template2)
   
input_text = "Create the bubble_sort algorithm in C#"
output_text = ""
for n in range(10): 
    out = ensemble(instruction=input_text, output=output_text)
    output_text += out
print(output_text)
