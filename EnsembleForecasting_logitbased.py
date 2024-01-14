# If the tokenizers are COMPLETELY identical (no added token) we can easily mean/max/... the logits. 
# TODO: adjust the function to work for a single model with two system prompts

def generate_completions(n, prompt):
    input_text = prompt
    response = ""
    display_text = ""
    generated_tokens_list = []

    for _ in range(n):
        clear_output(wait=True)
        input_text1 = prompt_template1.format(instruction=input_text, response=response)
        input_text2 = prompt_template2.format(instruction=input_text, response=response)

        input_ids1 = tokenizer1.encode(input_text1, return_tensors='pt').to("cuda")
        input_ids2 = tokenizer2.encode(input_text2, return_tensors='pt').to("cuda")

        with torch.no_grad():
            outputs1 = model1(input_ids=input_ids1)
            outputs2 = model2(input_ids=input_ids2)

        logits1 = outputs1.logits[:, -1, :]
        logits2 = outputs2.logits[:, -1, :]

        mean_logits = (logits1 + logits2) / 2
        max_logits, _ = torch.max(torch.stack((logits1, logits2)), dim=0)

        probs1 = torch.nn.functional.softmax(logits1, dim=-1)
        probs2 = torch.nn.functional.softmax(logits2, dim=-1)
        mean_logits = torch.nn.functional.softmax(mean_logits, dim=-1)
        max_logits =  torch.nn.functional.softmax(mean_logits, dim=-1)

        max_prob1, max_indices1 = torch.max(probs1, dim=-1)
        max_prob2, max_indices2 = torch.max(probs2, dim=-1)
        _, max_indices_mean_logits = torch.max(mean_logits, dim=-1)
        _, max_indices_max_logits = torch.max(max_logits, dim=-1)

        gen_token1 = tokenizer1.decode(max_indices1[0])
        gen_token2 = tokenizer2.decode(max_indices2[0])
        gen_token_mean_logits = tokenizer2.decode(max_indices_mean_logits[0])
        gen_token_max_logits = tokenizer2.decode(max_indices_max_logits[0])

        gen_token = gen_token_mean_logits #gen_token1 if max_prob1>max_prob2 else gen_token2
        if gen_token == gen_token1 == gen_token2: 
            color = '#000000' # darkmode'#93f5af'  # green
        elif gen_token == gen_token1:
            color = "#00a0fc" # Darkmode '#93c4f5'  # blue
        elif gen_token == gen_token2:
            color = '#e30508'  # pink
        else: 
            color = '#0000'
        
        if gen_token == "<|EOT|>":
            break
        if gen_token == " <｜end▁of▁sentence｜>":
            break
        response += gen_token
        html_token = gen_token.replace("\n", "<br>")
        display_text += f'<span style="color: {color}">{html_token}</span>'
        
        generated_tokens_list.append(gen_token)
        
        if response.endswith("\n\n\n"):
            break
        display(HTML(display_text.replace(r"\n", "<br>")))
        
        
        # print(f"| {tokenizer1.decode(generated_tokens1[0]):8s} | {probability1:.4f} | {probability1:.2%}")
        # print(f"| {tokenizer2.decode(generated_tokens2[0]):8s} | {probability2:.4f} | {probability2:.2%}")
    # print(response)
    return response
