from main import main
import numpy as np

for functional_intercept in [True, False]:
    for functional_params in [True, False]:
            main([
                "--functional_intercept", str(functional_intercept).lower(),
                "--functional_params", str(functional_params).lower(),
                "--model", "RUMBoost",
                "--device", "cuda",
                "--save_model", "true",
                "--learning_rate", "0.1",
                # "--lambda_l1", "0.01",
                # "--lambda_l2", "0.01",
                "--layer_sizes", "32",
                "--num_epochs", "500",
            ])