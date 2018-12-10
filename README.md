# deeplearning-project

## Naive Analysis
All initial results are here:
[source code](https://github.com/kylematoba/deeplearning-project/tree/master/code/SimpleAnalysis.ipynb)

## TPU Training with Wide ResNet
TPU training setup, currently configured for the best Naive Analysis policy:
[source code (Best Transforms as Sub-policy Policy)](https://github.com/kylematoba/deeplearning-project/blob/master/code/keras_wide_res_net_with_best_transforms_as_subpolicy.ipynb); [source code (AutoAugment Policy)](https://github.com/kylematoba/deeplearning-project/blob/master/code/keras_wide_res_net_source_code_aa.ipynb)
Note that the checkpoint system does not work with gmail accounts associated with @columbia.edu; a personal account must be used to enable the drive API.



## Reduced AutoAugment
Current version with support for category targeted transformations:
[source code](https://github.com/kylematoba/deeplearning-project/blob/master/code/AutoAugment_withPPO_withClasses.ipynb)
