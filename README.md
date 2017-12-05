# ML Project
## Project Description
+ To implement Machine Learning algorithms to estimate sentiment associated with words in a sentence. In this project, Viterbi algorithmn, Max marginal decoding and Viterbi with second order dependencies are used.

## Collaborators
+ Yin Ji Sheng(1001670)
+ Hilda Thian(1001776)
+ Teo Yang Rui(1001518)

## Instructions to run
### Before running:
+ Make sure you have Python 3
+ Make sure the `train`, `dev.out`, and `dev.in` are in the same directory as the `part2_3_4.py`

### To run:
1. Simply enter `python part2_3_4.py` in your terminal and you should observe new files `modified_train`,`dev.p2.out`, `dev.p3.out` and `dev.p4.out`
2. You can enter `python evalScript.py dev.out dev.p2.out` to evaluate the F scores of the output files
3. If you intende to change the input file, you will have to manually change it from the python script, and look for `load_original_train()` and `load_files()` functions and change the directory for train and dev.in files respectively. 

### Analysis of Accuracy for Part 5
+ Upon evaluation, it is observed that the score produced by Second-order Viterbi as an alternative is lower than the original Viterbi. This reduced score may be attributed to the fact that second-order viterbi is much stricter in its estimation as it now has to depend on an a sequence of two previous consecutive states instead of one. Using second-order Viterbi, there should be 7^3 possible transitions altogether since there are 7 different sentiment tags. However, there are only 56 transitions available from the training data. Given that the training set provided was very small and limited, the efficiency and accuracy of the second-order viterbi is largely restricted. Assuming that the limitations of the dataset outweighs the supposed increased level of accuracy in the second-order viterbi, we believe that the second-order viterbi is still better than the original. 
