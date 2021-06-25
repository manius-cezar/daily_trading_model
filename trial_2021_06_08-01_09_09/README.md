# Classification model for Cryptocurrency (BTCUSD)
In this directory you can find notebook with prepared model for Classification Model for Cryptocurrency to which I'm refering in my thesis.

# How to use?
Just simply run [`entry_model-categorical-crypto.ipynb`](./entry_model-categorical-crypto.ipynb) as you wish, you can change everything there according to your needs of discovering other model possibilities. If you want to change NN model implementation, do this in [`model_creation_categorical_crypto.py`](./model_creation_categorical_crypto.py) file. If you want to run the notebook, and have Tensorboard functionality, you will need to create first `my_dictionaries` directory in this localization, where logs from all trials you will perform, will be collected for further analysis.

# Metrics of the selected model after hyperparameters optimization
### Confusion matrix - train dataset
![Confusion matrix - train dataset](confusion_matrix_train.svg "Confusion matrix - train dataset")
### Confusion matrix - valid dataset
![Confusion matrix - valid dataset](confusion_matrix_valid.svg "Confusion matrix - valid dataset")
### Confusion matrix - test dataset
![Confusion matrix - test dataset](confusion_matrix_test.svg "Confusion matrix - test dataset")

### FN, FP, TN, TP
![FN](fn.svg "FN")
![FP](fp.svg "FP")
![TN](tn.svg "TN")
![TP](tp.svg "TP")

### Precision
![Precision](precision.svg "Precision")

### Recall
![Recall](recall.svg "Recall")

### Binary Crossentropy
![Binary Crossentropy](loss.svg "Binary Crossentropy")

### ROC
![ROC](roc_curve_all.svg "ROC")

### PRC Curve
![PRC Curve](prc_curve.svg "PRC Curve")

### AUC-ROC
![AUC-ROC](auc.svg "AUC-ROC")

### Binary Accuracy
![Binary Accuracy](binary_accuracy.svg "Binary Accuracy")