# Server code

There are 6 main scripts in the server module: 
- centralServer-dataset.py which is used for federated feature selection.
- centralServer-LR.py coordinates LR model training
- centralServer-SVM.py coordinates SVM model training
- centralServer-RF.py coordinates RF model training
- centralServer-ensemble.py coordinates ensemble classifier training (once all 3 model types have been trained)
- featureImportances.ipynb is used to extract the important features from feature selection or model training
