# Model Training  
The objective of this phase of the project is to train the LSTM model on the data generated in the previous stage before we proceed to final phase where the project will be deployed and will be used  
for optimizing instruction schedule using branch prediction.
# Files implemented  
This complete phase of the project has been implemented using *train_model.py* file.  
The outputs obtained are the model files with *.pkl* extension.  
# Important Dependencies used for training phase  
1. Tensorflow-keras
2. pickle
# Code overview  
Following are the functions implemented in the code:  
1. *_safe_eval_condition* : Used to parse expressions inside conditional statements of the code.
2. *_parse_for_iterations* : Used to parse expressions inside loop statements in the code.
3. *_extract_assignments_from_nodes* : Extracts assignment statements from a given set of nodes of a Control Flow Graph(CFG).
4.  *extract_training_df_from_dataset* : Used for encoding of the previously generated data to training data by storing branches and labels of each code.
5.  *tokenize_and_pad* : Performs tokenization and padding to training data.
6.  *build_and_train_model* : Trains personality based LSTM model on training data.
7.  *train_from_csv* : Main function of code. Calls the other functions to train the dataset
# Final Result  
After performing training of the data the final models are saved as binary files (.pkl) in the directory. They will be used in the later stages for making predictions.  
