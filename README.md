# unrolled-tbsca-ad
Code for reproducing the results of the paper "Adaptive Anomaly Detection in Network Flows with Deep Unrolling and Low-Rank Tensor Decompositions"

## Requirements
- Python 3.9+

| Package | Ver. |
| -------- | ---- |
| NumPy | 1.24.2+ |
| PyTorch | 1.13.1+ |
| Matplotlib | 3.7.1 |
| scikit-optimize | 0.10.1+ |
| PyWavelets | 1.4.1 |

## Description
The script ```script_paper_results.py``` runs all experiments done in the paper.
Inside the script, further sub-experiments can be activated and deactivated.

File directories (SCENARIODIR, RESULTDIR, EXPORTDIR, RW_ABILENE_DIR), CPU thread limit and device (CPU/GPU) can be configured in ```config.py```. The training and validation results are written to EXPORTDIR as CSV. Note that the memory optimization of the code is limited - it can easily exceed 20GB for some experiments.

## Usage
Paper reference will be added soon.

## References
- Y. Yang, M. Pesavento, Z.-Q. Luo and B. Ottersten, "Inexact Block Coordinate Descent Algorithms for Nonsmooth Nonconvex Optimization," in IEEE Transactions on Signal Processing, vol. 68, pp. 947-961, 2020, doi: 10.1109/TSP.2019.2959240. 
- H. Kasai, W. Kellerer and M. Kleinsteuber, "Network Volume Anomaly Detection and Identification in Large-Scale Networks Based on Online Time-Structured Traffic Tensor Tracking," in IEEE Transactions on Network and Service Management, vol. 13, no. 3, pp. 636-650, Sept. 2016, doi: 10.1109/TNSM.2016.2598788.
- M. Mardani, G. Mateos and G. B. Giannakis, "Dynamic Anomalography: Tracking Network Anomalies Via Sparsity and Low Rank," in IEEE Journal of Selected Topics in Signal Processing, vol. 7, no. 1, pp. 50-66, Feb. 2013, doi: 10.1109/JSTSP.2012.2233193.

## Known Issues
- For some newer PyTorch and Intel MKL versions, ```torch.linalg.lufactor``` produces erroneous pivots for certain input dimensions when limiting the number of CPU threads. This causes ```torch.linalg.solve``` to fail. Use GPU or deactivate thread limit in this case.
