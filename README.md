# Functional effects models: Accounting for preference heterogeneity in panel data with machine learning

Repository related to our paper on functional effects where we learn individual-specific intercepts and coefficients from the socio-demographic characteristics to account for inter-individual heterogeneity in panel data.

The case studies can be reproduced by running the [run_models.py](src/run_models.py) script, with correct arguments and dataset in the data folder.

The hyperparameter search can be reproduced by running the [hyperparameter_search.py](src/hyperparameter_search.py) python script.

The synthetic experiment can be reproduced by running the [synthetic_experiment.py](src/synthetic_experiment.py) python script.

The easySHare dataset pre-processing can be done by running the [data_preprocessing.ipynb](src/data_preprocessing.ipynb) jupyter notebook.

Finally, [models_wrapper.py](src/models_wrapper.py) contains all models wrapped in consistent classes from their source code.
