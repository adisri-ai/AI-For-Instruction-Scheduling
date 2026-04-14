# Overview  
This phase of the project deals with predicting the final complete flow of instructions for a given program and optimizing the cycle count accordingly.  
# Files for this phase  
The files created during this phase includeL  
1. *[instruction_scheduling.py](https://github.com/adisri-ai/AI-For-Instruction-Scheduling/blob/ff4550be7aba4b09199136c5454746020d2ddde8/flow_prediction/instruction_scheduling.py)* : Used for designing the Control Flow Graph of the user entered code and it's flow sequence. 
2. *[flow_inference_utils.py](https://github.com/adisri-ai/AI-For-Instruction-Scheduling/blob/14f4e5fbd91e0e3a3f4506ea738419df56f722e9/flow_prediction/flow_inference_utils.py)*   : Contrains functions for loading model files and using them for making predictions.
3. *pipeline_stimulator.py*    : Stimulates a pipeline of a CPU which is used for executing tasks in clock cycles. 
# Output from this phase  
For the given code these files are used to return the final instruction schedule after performing optimization. 
