# WDHT (UNDER UPDATE)
Implementation of Weakly Supervised Deep Image Hashing through Tag Embeddings

This repository is an implementation of the paper https://arxiv.org/abs/1806.05804. In the current implementation, we used ResNet50 of the Keras library instead of AlexNet as reported in the paper. This is due to unavailability of AlexNet model in Keras. Consequently, we are able to achieve slightly more accuaracy than reported in the paper. 

Installation:

1. This code is built using the following set-up
   - Python 2.7
   - Tensorflow 1.0.0
   - Keras 2.2.4
2. Replace the optimizers.py file in the Keras directory (/usr/local/lib/python2.7/dist-packages/keras) with the one in the current directory. The modified file contains a new class "FineTuneSGD" which is used in 12bits_NUS.py as an optimizer. 
3. Download the processed dataset from https://www.dropbox.com/sh/s5evhny6syk5q3m/AABHgjeUgWBa9Lnvi0XfstdGa?dl=0 and keep this folder at the same level as the code folder, WDHT (The data-set is processed to be in 'BGR' color order).
4. Download the pretrained weights from https://www.dropbox.com/sh/cqhd464glch2pkn/AADvSRMy8_6gP16rxnmNxSxta?dl=0 and keep this folder at the same level as the code folder, WDHT. 


Running:

a) Testing: Change the phase to 'Testing' in the "12bits_NUS.py" code and directly obtain the mean average precision. The pretrained weights are already downloaded as a part of the installation process above. 
b) Training: Change the phase to 'Training' to train the network as given in the paper. 


P.S 
Please contact me at vijetha.gattupalli@gmail.com for any questions, concerns or bugs. 
