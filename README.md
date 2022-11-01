# Code for federated DCI detection classifier
Code associated with paper titled 'A generalizable physiological model for detection of delayed cerebral ischemia using federated learning'. The code is split into three modules:
- Server module which coordinates FL learning and aggregates model weights from local sites. We develop a novel federated feature selection process and employ FedAvg for optimization.
- Client module which trains the model on local data. The module can be used by each client separately and only requires input of the directory where the dataset is stored.
- Helper module with code taken from Megjhani et al., 2021, Stroke. This code is used to create the same features as used in their paper.
