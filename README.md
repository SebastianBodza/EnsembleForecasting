# EnsembleForecasting
Using multiple LLMs for ensemble Forecasting. 

Got the Idea from participation in a Kaggle Challenges with Tabular-Data and TimeSeries-Data. Showing massive performance boost from Ensembling multiple Models with simple averaging of the results. Especially in uncertainty! Why not also for LLMs? 

First unoptized proof of concept with two quantized (AWQ) Models. Can also be run on Colab!

**tl;dr:**
With an Ensemble of two 4-bit quanitzed Models it is possible to beat the official Human-Eval Scores. 


# Results
## Humaneval
|   all AWQ quantization     | Magicode | Deepseek | Ensemble TT [1] | Ensemble MT [2] | Ensemble MinT [3] |
|-----------|--------------|--------------|----------|----------|----------|
| ~ 7B | 71.95%       | 76.83%       | 77.44%   | 76.83%   | 76.22%   |


|   all AWQ quantization     | Phind-34B-AWQ | Deepseek-33B-AWQ | Ensemble TT [1] | 
|-----------|--------------|--------------|----------|
| ~ 34B | 74.39%       |    78.05%   | 79.89%   | 

[1] Taking the token with the higher probability after taking the max   
[2] Taking the average of the logits and the sampling the highest probability  
[3] Taking the min of both outputted logits and the sampling the highest probability  

# Next Step: 
Ensemble on Logprob base: 
- Take the maximum of both -> just sanity checking [1]
  
Others:
- using the same model with different system prompts (idea from x @nanulled) e.g. you are debugging and you are a code generator -> easier to implement with better throughput 
- Ensemble LoRa serving with S-LoRa
- Taking more diverse models -> not too much variance in the models and deepseek seems to be fairly dominant.

# Updates: 
- 2024-01-14: Added Results for single models and ensemble
- 2024-01-16: Added Results for 34B Models beating the official Score from Deepseek with AWQ-Models (4-bit quantization)
