from main import main

for functional_intercept in [True, False]:#, False]:#, 
    for functional_params in [True, False]:#, False]:#True, 
        for dataset in ["LPMC", "SwissMetro", "easySHARE"]: #, "LPMC"]:#, "easySHARE"]:#"]: "SwissMetro", 
            for model in ["TasteNet", "DNN"]:#"TasteNet"]:#, "RUMBoost"

                if model == "DNN" and functional_params:
                    continue

                main([
                    "--functional_intercept", str(functional_intercept).lower(),
                    "--functional_params", str(functional_params).lower(),
                    "--model", model,
                    "--save_model", "true",
                    "--optimal_hyperparams", "true",
                    "--dataset", dataset,
                ])