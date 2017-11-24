# HPSDE

This is the matlab code for Hierarchical Policy Search via Return-Weighted Density Estimation (HPSDE).

Dependencies:
minConf by Mark Schmidt
https://www.cs.ubc.ca/~schmidtm/Software/minConf.html

Download the zip file and locate the folder "minConf" in the same folder as this readme.md

gpml-matlab by Carl Edward Rasmussen and Hannes Nickisch
http://www.gaussianprocess.org/gpml/code/matlab/doc/

Download the zip file and locate the folder "gpml-matlab" in the same folder as this readme.md

How to run:
To see the toy tasks in the paper, run
demo_HPSDE_toy
You can switch the return functions by changing 'RewardType' in the script.

To see the toy tasks in the paper, run
demo_HPSDE_puddle
You can switch the return functions by changing 'RewardType' in the script.
It takes about 30min to finish the learning using a laptop with Intel Core i7-6500U CPU.

If you want to skip the learning and see the performance of a pre-trained hierarchical policy, run
VisualizeSavedPolicy_puddle

To see the behavior of the return-weighted density estimation with VBEM, run
demo_ReturnWeightedClustering




