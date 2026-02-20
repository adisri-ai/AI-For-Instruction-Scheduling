# Objective
1. The first step of the project includes generating randomized C++ codes having branches in their Control Flow Graph(CFG) such as loops or conditionals.
# Concerend File  
1. The code in the file *training.py* does this job and the final training data is saved in the file *data.csv*
# Personality based Code Generation
1. There are 4 personality classes taken into consideration for this project:
   1. **Verbose coder**
   2. **Compact coder**
   3. **Experimental coder**
   4. **Structured coder**
2. The file *training.py* consists of the following functions:
    1. *_rand_var()* : Used for generating random indentifiers
    2. *_assign_statement* : To generate statements involing the assignment(=) operator
    3. *_if_block_chain* : Generate random if-block chain in the code along with label of True/False value of the conditional
    4. *_for_block*  : Generation of for-block with a label indicating whether the loop executed.
       
