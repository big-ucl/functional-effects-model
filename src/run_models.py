from main import main
import numpy as np

for functional_intercept in [True, False]:#, False]:#, 
    for functional_params in [True, False]:#, False]:#True, 
        for dataset in ["LPMC"]:#, "easySHARE"]:#"]: "SwissMetro", 
            for model in ["RUMBoost", "TasteNet"]:#"TasteNet"]:#, 

                main([
                    "--functional_intercept", str(functional_intercept).lower(),
                    "--functional_params", str(functional_params).lower(),
                    "--model", model,
                    "--save_model", "true",
                    "--optimal_hyperparams", "true",
                    "--dataset", dataset,
                ])