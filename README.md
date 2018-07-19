# WDHT
Implementation of Weakly Supervised Deep Image Hashing through Tag Embeddings


This repository is an implementation of the paper https://arxiv.org/abs/1806.05804


Installation:

1. Install Keras ( https://keras.io/ ) with theano backend. 
2. Replace the optimizers.py file in the Keras directory with the one in the current directory. The modified file contains a new class "FineTuneSGD" which is used in 12bits_NUS.py as an optimizer. 
3. Download the processed dataset from https://www.dropbox.com/sh/s5evhny6syk5q3m/AABHgjeUgWBa9Lnvi0XfstdGa?dl=0 and keep this folder at the same level as the code folder, WDHT.
4. Download the pretrained weights from https://www.dropbox.com/sh/cqhd464glch2pkn/AADvSRMy8_6gP16rxnmNxSxta?dl=0 and keep this folder at the same level as the code folder, WDHT. 


Running:

a) Testing: Change the phase to 'Testing' and directly obtain the mean average precsion by running the "12bits_NUS.py" code. The pretrained weights are already downloaded as a part of the installation process above. 
b) Training: Change the phase to 'Training' to train the network for as given in the paper. 


P.S 
Please contact me at vijetha.gattupalli@asu.edu for any questions, concerns or bugs. 
