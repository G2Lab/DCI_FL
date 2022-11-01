# Associated code for federated DCI detection classifier
Code associated with paper titled 'A generalizable physiological model for detection of delayed cerebral ischemia using federated learning'. The code is split into three modules:
- Server module which coordinates FL learning and aggregated model weights. We have a federated feature selection process and employ FedAvg for optimization.
- Client module which trains the model on local data. The module can be used by each separately and just requires input of the directory for the dataset.
- Helper module with code taken from Megjhani et al., 2021, Stroke. This code is used to create the same features as used in their paper.
