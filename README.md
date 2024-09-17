# unrolled-tbsca-ad
Code for reproducing the results of the paper "Adaptive Anomaly Detection in Network Flows with Low-Rank Tensor Decompositions and Deep Unrolling"

## Requirements
- Python 3.8+

| Package | Ver. |
| -------- | ---- |
| NumPy | 1.24.2+ |
| PyTorch | 1.13.1+ |
| Matplotlib | 3.7.1 |
| scikit-optimize | 0.10.1+ |
| PyWavelets | 1.4.1 |

## Description
### Executing training and validation
The script ```script_paper_results.py``` runs all experiments done in the paper. Arguments: ```EXP --cvs CVS [--parallel]```
1) ```EXP```: There are 5 experiments that can be run:
	- ```synth_ab```: Runs the script that was used to perform the feature distillation for parameters **W** and **M**. The distillation by model comparison is not automatic, and experiments are by progressively adding the feature group yielding the best performance and iterating through the remaining groups. The progression is indicated by the ```Round X``` comments.
	- ```synth_comp```: Runs comparisons of different methods on synthetic data.
	- ```abil_comp```: Runs comparisons of different methods on Abilene data.
	- ```class_comp```: Runs comparisons on "classical" (not unrolled) algorithms on a synthetic data set using Bayesian optimization methods.
	- ```custom```: Runs a training run and validation for arbitrary parameters. The experimental parameters are described and set within the script. 
2) ```--cvs CVS```: list of cross-validation split indices, e.g., 0 1 2 3 4. ```synth_abl```, ```synth_comp```, ```class_comp``` and ```custom``` (per default) have 5 splits in total (0 1 2 3 4). ```abil_comp``` has 4 splits.
3) ```--parallel```: optional, splits will be run simultaneously. Only supported for CPU.

Inside the script, further sub-experiments for ```synth_abl```, ```synth_comp``` and ```abil_comp``` can be activated and deactivated. Note that single simulations, of which there are multiple per EXP, may take many hours for each split.

File directories (```SCENARIODIR```, ```RESULTDIR```, ```EXPORTDIR```, ```RW_ABILENE_ROUTINGTABLE_PATH```, ```RW_ABILENE_FLOW_PATH```), CPU thread limit and device (CPU/GPU) can be configured in ```config.py```. Note that the memory optimization of the code is limited - it can easily exceed 20GB for some experiments.

### Models and Algorithms
Classical algorithms: 
- ```bbcd``` (from Mardani et al.)
- ```bsca``` (from our prev. conference publication)
- ```bsca_tens_nrlx``` (Alg. 1)
- ```bsca_tens_rlx``` (Alg. 2)

Learning-based architectures: 
- ```BSCATensorUnrolled``` (proposed architecture, see ```script_paper_results.py``` for configuration)
- ```BSCAUnrolled``` (from our prev. conference publication)

### Data and Results
Synthetic data sets and processed RW data are stored in ```SCENARIODIR```.
Trained models and results are pickled and stored in ```RESULTDIR```.
The results are further exported as tabular data into ```EXPORTDIR```. They are named as follows:
- ```bayopt_{ALG_NAME}_ON_{VALIDATION_DATA_NAME}_cvs{INT}.txt```: Results in AUC for classical method for one data split for best iteration number or over all layers.
- ```bayopt_{ALG_NAME}_{VALIDATION_DATA_NAME}_cvs{INT}_iter.txt```: Results in AUC for classical method for one data split over all iterations.
- ```{RUN_NAME}_cvs{SPLIT}{ACR}_ON_{VALIDATION_DATA_NAME}.txt```: Results in AUC for learning-based methods for one data split over all layers.
- ```{RUN_NAME}_cvs{SPLIT}{ACR}_tloss.txt```: Training loss for learning-based methods for one data split over training steps.
- ```{RUN_NAME}_cvs{SPLIT}{ACR}_tepochs.txt```: Validation AUC and average regularization parameters for learning-based methods for one data split over training epochs.

```RUN_NAME``` specifies the learning-based architecture, its configuration and the data set used to train the weights.
```ACR``` specifies a variation of the run parameters (except model parameters), e.g., ```_lfss{SUBSAMPLING_FACTOR}``` describes the subsampling factor of the loss function, ```_tsetsz{TRAINING_SET_SIZE}``` for a reduced-size training set.

## Abilene Dataset
The Abilene realworld dataset can be downloaded from, e.g., [https://www.cs.utexas.edu/~yzhang/research/AbileneTM/](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/) (Zhang et al. 2003 - "Fast Accurate Computation of Large-Scale IP Trafﬁc Matrices from Link Loads").
We need the routing table ```A``` (set RW_ABILENE_ROUTINGTABLE_PATH accordingly) and ```X{NUM}.gz``` (set RW_ABILENE_FLOW_PATH accordingly.)

## Usage
Please cite L. Schynol and M. Pesavento, "Adaptive Anomaly Detection in Network Flows with Low-Rank Tensor Decompositions and Deep Unrolling" if you apply the provided code in you own work. If you use the implementations of the other algorithms, please cite the respective works as well.

## References
- Y. Yang, M. Pesavento, Z.-Q. Luo and B. Ottersten, "Inexact Block Coordinate Descent Algorithms for Nonsmooth Nonconvex Optimization," in IEEE Transactions on Signal Processing, vol. 68, pp. 947-961, 2020, doi: 10.1109/TSP.2019.2959240. 
- H. Kasai, W. Kellerer and M. Kleinsteuber, "Network Volume Anomaly Detection and Identification in Large-Scale Networks Based on Online Time-Structured Traffic Tensor Tracking," in IEEE Transactions on Network and Service Management, vol. 13, no. 3, pp. 636-650, Sept. 2016, doi: 10.1109/TNSM.2016.2598788.
- M. Mardani, G. Mateos and G. B. Giannakis, "Dynamic Anomalography: Tracking Network Anomalies Via Sparsity and Low Rank," in IEEE Journal of Selected Topics in Signal Processing, vol. 7, no. 1, pp. 50-66, Feb. 2013, doi: 10.1109/JSTSP.2012.2233193.

## Known Issues
- For some newer PyTorch and Intel MKL versions, ```torch.linalg.lufactor``` produces erroneous pivots for certain input dimensions when limiting the number of CPU threads. This causes ```torch.linalg.solve``` to fail. Use GPU or deactivate thread limit in this case.
