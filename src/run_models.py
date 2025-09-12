from main import main
import numpy as np

for functional_intercept in [False]:#, False]:#, 
    for functional_params in [False]:#, False]:#True, 
        for dataset in ["LPMC", "SwissMetro"]: #, "LPMC"]:#, "easySHARE"]:#"]: "SwissMetro", 
            for model in ["GBDT", "DNN"]:#"TasteNet"]:#, "RUMBoost"

                main([
                    "--functional_intercept", str(functional_intercept).lower(),
                    "--functional_params", str(functional_params).lower(),
                    "--model", model,
                    "--save_model", "true",
                    "--optimal_hyperparams", "true",
                    "--dataset", dataset,
                ])