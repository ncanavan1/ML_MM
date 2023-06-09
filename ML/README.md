Use train_model.py to train a ML model. The file sample.csv contains a sample of a reduced features space with associated U values for a range of moleceules.

By submitting the job file via slurm or by executing ./train_model.py and passing as input the features file and either 2 (regression model) or 3 (classification model). 1 is reserved for a regression model for predicitong a molecules minimum energy.

Output will show final performance for model created in leanrning curves and either a scatter plot or confusion matrix.
