from main import main
import numpy as np

for functional_intercept in [True]:#, False]:#, 
    for functional_params in [True]:#, False]:#True, 
        for dataset in ["easySHARE"]:#"]: "SwissMetro", 
            for model in ["TasteNet"]:#"TasteNet"]:#, 
                if not functional_intercept and not functional_params:
                    continue
                main([
                    "--functional_intercept", str(functional_intercept).lower(),
                    "--functional_params", str(functional_params).lower(),
                    "--model", model,
                    "--save_model", "true",
                    "--optimal_hyperparams", "true",
                    "--dataset", dataset,
                ])