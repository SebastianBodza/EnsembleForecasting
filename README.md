# EnsembleForecasting
Using multiple LLMs for ensemble Forecasting. 

Got the Idea from participation in a Kaggle Challenges with Tabular-Data and TimeSeries-Data. Showing massive performance boost from Ensembling multiple Models with simple averaging of the results. Especially in uncertainty! Why not also for LLMs? 

First unoptized proof of concept with two quantized (AWQ) Models. Can also be run on Colab!

Next Step: Ensemble on Logprob base: 
- Taking the average from both
- Take the maximum of both
- using the same model with different system prompts (idea from x @nanulled) e.g. you are debugging and you are a code generator -> easier to implement with better throughput 
