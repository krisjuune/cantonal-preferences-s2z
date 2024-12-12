import pymc as pm 
import pandas as pd
import arviz as az
import xarray as xr
from functions.data_assist import apply_mapping

# %% pymc bug workaround

import pytensor
pytensor.config.cxx = '/usr/bin/clang++'

# %% import data

df_pv = pd.read_csv("data/pv-conjoint.csv")
df_heat = pd.read_csv("data/heat-conjoint.csv")

# %% define lists and translations

attributes_pv = [ 
    "mix", 
    "imports",
    "pv",
    "tradeoffs", 
    "distribution"
]

attributes_heat = [
    "year",
    "tax",
    "ban",
    "heatpump",
    "energyclass",
    "exemption"
]

baselines_pv = [
    "mix:hydro",
    "imports:0%",
    "pv:none", 
    "tradeoffs:none",
    "distribution:none"
]

baselines_heat = [
    "year:2050", 
    "tax:0%",
    "ban:none",
    "heatpump:subsidy",
    "energyclass:new-only-efficient",
    "exemption:none"
]

translate_dict_pv = {
    # target mix
    'More hydro': 'hydro',
    'More solar': 'solar',
    'More wind': 'wind',

    # rooftop pv requirements
    'No obligation': 'none',
    'New public and commercial buildings': 'new-non-residential',
    'New and existing public and commercial buildings': 'all-non-residential',
    'All new buildings': 'all-new',
    'All new and existing buildings': 'all',

    # biodiversity tradeoffs
    'No trade-offs': 'none',
    'Alpine regions': 'alpine',
    'Agricultural areas': 'agricultural',
    'Forests': 'forests',
    'Rivers': 'rivers',
    'Lakes': 'lakes',

    # cantonal distribution
    'No agreed distribution': 'none',
    'Potential-based': 'potential-based',
    'Equal per person': 'equal-pp', 
    'Minimum limit': 'min-limit',
    'Maximum limit': 'max-limit',
}

translate_dict_heat = {
    # ban
    'No ban': 'none',
    'Ban on new installations': 'new',
    'Ban and fossil heating replacement': 'all',

    # heatpump
    'Subsidy': 'subsidy',
    'Governmental lease': 'lease',
    'Subscription': 'subscription',

    # energyclass 
    'New buildings must be energy efficient': 'new-only-efficient',
    'New buildings must be energy efficient and produce renewable electricity on-site': 'new-efficient-renewable',
    'All buildings must be energy efficient': 'all-retrofit', 
    'All buildings must be energy efficient and produce renewable electricity on-site': 'all-retrofit-renewable',

    # exemptions
    'No exemptions': 'none',
    'Low-income households are exempted': 'low',
    'Low and middle-income households are exempted': 'low-mid'
}

# %% define dummies

df = df_pv
translate_dict = translate_dict_pv
attributes = attributes_pv
baselines = baselines_pv

df = apply_mapping(df, translate_dict)

# set baselines
baseline_dict = {attr.split(":")[0]: attr.split(":")[1] for attr in baselines}

# Reorder each attribute column by making it categorical with the baseline first
for attr in attributes:
    baseline = baseline_dict[attr]
    df[attr] = pd.Categorical(df[attr], categories=[baseline] + 
                              [level for level in df[attr].unique() if level != baseline], 
                              ordered=True)

# Generate dummies with columns in the correct order
dummies = pd.get_dummies(df[attributes], drop_first=False)

# Reorder columns to place baseline first for each attribute
ordered_columns = []
for attr in attributes:
    # Collect the columns related to the attribute and put baseline first
    attr_columns = [col for col in dummies.columns if col.startswith(attr)]
    baseline_column = f"{attr}_{baseline_dict[attr]}"
    ordered_columns.append(baseline_column)
    ordered_columns.extend([col for col in attr_columns if col != baseline_column])

# Reorder dummies according to ordered columns list
dummies = dummies[ordered_columns]
dummies = dummies.loc[:, ~dummies.columns.duplicated()]

df["canton"] = df["canton"].astype("category")

#TODO add task dimension but doesn't yet exist in the data maybe add in the dataframe itself
coords = {"level": dummies.columns.values, 
          "canton": df["canton"].cat.categories,}

# %% build model

with pm.Model(coords = coords) as bayes_model: 
    beta_mean = pm.Normal("beta_mean", 0, sigma = 2, dims = "level")

    canton_mean = pm.Normal(
        "canton_mean", 
        0, 
        sigma = 1, 
        dims = ["canton", "level"])
    
    canton_sigma = pm.Exponential(
        "canton_sigma", 
        1, 
        dims = "level")

    canton_effect = pm.Deterministic(
        "canton_effect", 
        canton_mean * canton_sigma,
        dims = ["canton", "level"])
    
    # column of which canton index per task
    c = pm.Data(
        "c", 
        df.loc[df.pack_num_cat == "Left", "canton"].cat.codes, 
        dims = "task"
    )

    beta = pm.Deterministic(
        "beta", 
        beta_mean + canton_effect, 
        dims = ["canton", "level"]
    )

    observed_choice_left = pm.Data(
        "observed_choice_left", 
        df.loc[df.pack_num_cat == "Left", "Y"].values, 
        dims = ["task"]
    )

    attribute_levels_left = pm.Data(
        "attribute_levels_left", 
        dummies[df.pack_num_cat == "Left"].values, 
        dims = ["task", "level"])
    
    utility_left = pm.Deterministic(
        "utility_left",
        pm.math.sum(attribute_levels_left * beta[c, :], axis = 1), 
        dims = "task")
    
    attribute_levels_right = pm.Data(
        "attribute_levels_right", 
        dummies[df.pack_num_cat == "Right"].values, 
        dims = ["task", "level"])
    
    utility_right = pm.Deterministic(
        "utility_right",
        pm.math.sum(attribute_levels_right * beta[c, :], axis = 1), 
        dims = "task")
    
    probability_choice_left = pm.Deterministic(
        "probability_choice_left", 
        pm.math.exp(utility_left)/(pm.math.exp(utility_left)+pm.math.exp(utility_right)))

    choice_distribution = pm.Bernoulli(
        "choice_distirbution",
        p = probability_choice_left, 
        observed = observed_choice_left)

    

# %% get priors

priors = pm.sample_prior_predictive(
    samples = 1000, 
    model = bayes_model, 
    random_seed = 42, 
)

# %% check priors
az.summary(priors, var_names = ["canton_sigma"])

# %% run model with MCMC

# run model with MCMC with 1000 draws, 500 tune samples, and 4 chains
inference_data = pm.sample(
    model = bayes_model, 
    draws = 1000, 
    tune = 500, 
    cores = 4, 
    random_seed = 42, 
    return_inferencedata = True, 
    target_accept = 0.9
)

#TODO you can check the meaning by using the Pr = ... equation (and setting right to 0)

# %% diagnostics

inference_data["posterior"]["canton_effect"].mean(["chain", "draw"]).max(["canton"])
az.plot_trace(inference_data, var_names = ["beta_mean", "canton_sigma", "beta"])

# %%
