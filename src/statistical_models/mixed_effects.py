import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.models import logit
from biogeme.expressions import Beta, PanelLikelihoodTrajectory, Draws, log, MonteCarlo
import pandas as pd


def define_and_return_biogeme(
    df: pd.DataFrame, alt_spec_vars: dict[int, list[str]], num_classes: int
) -> bio.BIOGEME:
    """
    Define and return a Biogeme object for fixed effects models.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data.
    alt_spec_vars: dict[int, list[str]]
        Dictionary mapping alternative IDs to lists of variable names.
    """
    database = db.Database("fixed_effects_data", df)
    database.panel("ID")

    betas = define_betas(alt_spec_vars)

    CHOICE = database.variables["CHOICE"]

    ascs = {}

    for i in range(num_classes):
        ascs[i] = Beta(f"asc_{i}", 0, None, None, i == num_classes - 1) + Beta(
            f"mu_{i}", 1, None, None, i == num_classes - 1
        ) * Draws("draws", "MLHS")

    V = {}
    for alt_id, vars in alt_spec_vars.items():
        V[alt_id] = (
            sum(
                betas[f"beta_{var}_alt{alt_id}"] * database.variables[var]
                for var in vars
            )
            + ascs[alt_id]
        )

    av = {alt_id: 1 for alt_id in V.keys()}

    choice_probability_one_observation = logit(V, av, CHOICE)

    conditional_trajectory_probability = PanelLikelihoodTrajectory(
        choice_probability_one_observation
    )

    log_probability = log(MonteCarlo(conditional_trajectory_probability))

    the_biogeme = bio.BIOGEME(database, log_probability, number_of_draws=500, seed=0)

    return (
        the_biogeme,
        log_probability,
        conditional_trajectory_probability,
        ascs,
        database,
    )


def define_betas(alt_spec_vars: dict[int, list[str]]) -> dict[str, Beta]:
    """
    Define beta parameters for the fixed effects model.

    Parameters
    ----------
    alt_spec_vars: dict[int, list[str]]
        Dictionary mapping alternative IDs to lists of variable names.

    Returns
    -------
    dict[str, Beta]
        Dictionary of beta parameters.
    """
    betas = {}
    for alt_id, vars in alt_spec_vars.items():
        for var in vars:
            beta_name = f"beta_{var}_alt{alt_id}"
            if beta_name not in betas:
                betas[beta_name] = Beta(beta_name, 0, None, None, 0)
    return betas
