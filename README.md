# deepgemm

you must download and use DeepGEMM library: https://github.com/deepseek-ai/DeepGEMM/tree/main 
### Requirements
Hopper architecture GPUs, sm_90a must be supported
Python 3.8 or above
CUDA 12.3 or above
But we highly recommend 12.8 or above for the best performance
PyTorch 2.1 or above
CUTLASS 3.6 or above (could be cloned by Git submodule)
### Submodule must be cloned
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
python setup.py install
