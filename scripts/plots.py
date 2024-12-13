import altair as alt
import geopandas as gpd
from shapely.ops import transform
import pyproj
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# %% get pathworth utilities

# cantonal variability (gamma)
cantonal_beta = inference_data["posterior"]["canton_effect"].mean(["chain", "draw"])
cantonal_beta = cantonal_beta.to_dataframe(name="cantonal_beta").reset_index()

# national means (alpha)
beta_mean = inference_data["posterior"]["beta_mean"].mean(["chain", "draw"])
beta_mean = beta_mean.to_dataframe(name="beta_mean").reset_index()
cantonal_beta = cantonal_beta.merge(beta_mean, 
                                    on = "level", 
                                    how = "left")

# add total pathworth utilities (canton dependent)
cantonal_beta["beta"] = cantonal_beta["cantonal_beta"] + cantonal_beta["beta_mean"]

# %% define and choose order 

# attribute level order for plotting
desired_order_pv = [
    "mix_hydro", 
    "mix_solar", 
    "mix_wind",
    "imports_0%",
    "imports_10%",
    "imports_20%",
    "imports_30%",
    "pv_none",
    "pv_new-non-residential",
    "pv_all-new",
    "pv_retrofit-non-residential",
    "pv_retrofit-all",
    "tradeoffs_none",
    "tradeoffs_alpine",
    "tradeoffs_agriculural",
    "tradeoffs_lakes",
    "tradeoffs_rivers",
    "tradeoffs_forests",
    "distribution_none",
    "distribution_potential-based",
    "distribution_equal-pp",
    "distribution_min-limit",
    "distribution_max-limit"
    ]

desired_order_heat = [
    "year_2050",
    "year_2045",
    "year_2040",
    "year_2035",
    "year_2030",
    "tax_0%",
    "tax_25%",
    "tax_50%",
    "tax_75%",
    "tax_100%",
    "ban_none",
    "ban_new",
    "ban_all",
    "heatpump_subsidy", 
    "heatpump_lease", 
    "heatpump_subscription",
    "energyclass_new-only-efficient",
    "energyclass_new-efficient-renewable", 
    "energyclass_all-retrofit",
    "energyclass_all-retrofit-renewable",
    "exemptions_none",
    "exemptions_low",
    "exemptions_low-mid"
]

desired_order = desired_order_heat

# %% get cantonal boundaries

# Load shapefile from https://www.swisstopo.admin.ch/de/landschaftsmodell-swissboundaries3d
cantons_gdf = gpd.read_file("data/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp", 
                            engine = "pyogrio")

# Ensure that the data is projected to WGS84 (lat/lon)
cantons_gdf = cantons_gdf.to_crs(epsg=4326)

# Convert to GeoJSON and save
cantons_gdf.to_file("data/swiss_cantons.geojson", 
                    driver="GeoJSON", 
                    engine = "pyogrio")

with open("data/swiss_cantons.geojson") as f:
    cantons_geojson = json.load(f)

# Define coordinate systems for transformation from Swiss LV03 to WGS84
lv03 = pyproj.CRS('EPSG:21781') 
wgs84 = pyproj.CRS('EPSG:4326')  
project = pyproj.Transformer.from_crs(lv03, wgs84, always_xy=True).transform

# Load the GeoJSON with pyogrio (or with geopandas for quick testing)
cantons = gpd.read_file('data/swiss_cantons.geojson')

# Transform geometries to WGS84
cantons['geometry'] = cantons['geometry'].apply(lambda geom: transform(project, geom) if geom is not None else None)

# %% test map

# Load GeoJSON file for Swiss cantons in WGS84
cantons = gpd.read_file("data/swiss_cantons.geojson")

# Add a column with random values to simulate data
cantons["random_value"] = np.random.uniform(-1, 1, size=len(cantons))

# Plot using the random values with a red-to-blue color scheme
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cantons.plot(column='random_value', cmap='coolwarm', legend=True, ax=ax)

# Add title and display
plt.title("Swiss Cantons with Random Values (Red to Blue)")
plt.show()


# %% define levels for maps

# filter for specific attribute levels
# pv
beta_potential = cantonal_beta[cantonal_beta['level'] == "distribution_potential-based"]
beta_equal = cantonal_beta[cantonal_beta['level'] == "distribution_equal-pp"]
beta_min = cantonal_beta[cantonal_beta['level'] == "distribution_min-limit"]
beta_max = cantonal_beta[cantonal_beta['level'] == "distribution_max-limit"]
beta_nodistribution = cantonal_beta[cantonal_beta['level'] == "distribution_none"]

beta_alpine = cantonal_beta[cantonal_beta['level'] == "tradeoffs_alpine"]
beta_lakes = cantonal_beta[cantonal_beta['level'] == "tradeoffs_lakes"]
beta_forests = cantonal_beta[cantonal_beta['level'] == "tradeoffs_lakes"]
beta_notradeoffs = cantonal_beta[cantonal_beta['level'] == "tradeoffs_none"]

beta_0imports = cantonal_beta[cantonal_beta['level'] == "imports_0%"]
beta_10imports = cantonal_beta[cantonal_beta['level'] == "imports_10%"]
beta_20imports = cantonal_beta[cantonal_beta['level'] == "imports_20%"]
beta_30imports = cantonal_beta[cantonal_beta['level'] == "imports_30%"]

# heat
beta_2030 = cantonal_beta[cantonal_beta['level'] == "year_2030"]
beta_2035 = cantonal_beta[cantonal_beta['level'] == "year_2035"]
beta_2040 = cantonal_beta[cantonal_beta['level'] == "year_2040"]
beta_2045 = cantonal_beta[cantonal_beta['level'] == "year_2045"]
beta_2050 = cantonal_beta[cantonal_beta['level'] == "year_2050"]

beta_0tax = cantonal_beta[cantonal_beta['level'] == "tax_0%"]
beta_25tax = cantonal_beta[cantonal_beta['level'] == "tax_25%"]
beta_50tax = cantonal_beta[cantonal_beta['level'] == "tax_50%"]
beta_75tax = cantonal_beta[cantonal_beta['level'] == "tax_75%"]
beta_100tax = cantonal_beta[cantonal_beta['level'] == "tax_100%"]

beta_subsidy = cantonal_beta[cantonal_beta['level'] == "heatpump_subsidy"]
beta_lease = cantonal_beta[cantonal_beta['level'] == "heatpump_lease"]
beta_subscription = cantonal_beta[cantonal_beta['level'] == "heatpump_subscription"]

beta_noban = cantonal_beta[cantonal_beta['level'] == "ban_none"]
beta_newban = cantonal_beta[cantonal_beta['level'] == "ban_new"]
beta_allban = cantonal_beta[cantonal_beta['level'] == "ban_all"]

beta_noexemption = cantonal_beta[cantonal_beta['level'] == "exemption_none"]
beta_low = cantonal_beta[cantonal_beta['level'] == "exemption_low"]
beta_lowmid = cantonal_beta[cantonal_beta['level'] == "exemption_low-mid"]

# Define levels and data for each map

# pv
levels_distribution = {
    "distribution_none": beta_nodistribution,
    "distribution_potential-based": beta_potential,
    "distribution_equal-pp": beta_equal,
    "distribution_max-limit": beta_max
}

levels_tradeoffs = {
    "tradeoffs_none": beta_notradeoffs,
    "tradeoffs_forests": beta_forests,
    "tradeoffs_alpine": beta_alpine,
    "tradeoffs_lakes": beta_lakes
}

levels_imports = {
    "imports_0%": beta_0imports, 
    "imports_10%": beta_10imports, 
    "imports_20%": beta_20imports, 
    "imports_30%": beta_30imports
}

# heat
levels_year = {
    "year_2030": beta_2030,
    "year_2040": beta_2040,
    "year_2050": beta_2050,
}

levels_tax = {
    "tax_0%": beta_0tax,
    "tax_50%": beta_50tax,
    "tax_100%": beta_100tax,
}

levels_ban = {
    "ban_none": beta_noban,
    "ban_new": beta_newban,
    "ban_all": beta_allban,
}

levels_exemption = {
    "exemption_none": beta_noexemption,
    "exemption_low": beta_low,
    "exemption_low-mid": beta_lowmid,
}


# %% plotting attribute levels on map

# Define a function to plot maps and save them
def plot_cantonal_beta_map(levels_dict, filename_suffix, cmap=plt.cm.coolwarm.reversed(), vmin=-0.4, vmax=None):
    """
    Plot maps for each level in levels_dict and saves the figure with a given filename_suffix.

    Parameters:
        levels_dict (dict): Dictionary containing level names as keys and dataframes as values.
        filename_suffix (str): Prefix for the saved figure files.
        cmap (Colormap): Colormap for the plot. Default is reversed coolwarm.
        vmin (float): Minimum value for normalization. Default is -0.4.
        vmax (float): Maximum value for normalization. Default is computed from data.
    """
    # compute vmin and vmax if not provided
    if vmin is None:
        vmin = min(df['beta'].min() for df in levels_dict.values())

    if vmax is None:
        vmax = max(df['beta'].max() for df in levels_dict.values())

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    num_levels = len(levels_dict)
    if num_levels <= 3:
        rows = 1
        cols = num_levels 
    else:
        rows = 2  
        cols = (num_levels + 1) // 2 if num_levels % 2 != 0 else num_levels // 2 
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 6), constrained_layout=True)

    # flatten axes for easier iteration, if only one row adjust to single dimension
    if rows == 1:
        axes = np.array(axes)

    # Iterate over levels and plot each map
    for ax, (level_name, beta_level) in zip(axes.flat, levels_dict.items()):
        # Merge data for the specific level
        merged_df = cantons.merge(beta_level, left_on="NAME", right_on="canton", how="left")

        # Plot map
        merged_df.plot(column='beta', cmap=cmap, legend=False, ax=ax, norm=norm)
        ax.set_title(level_name.replace("_", " ").capitalize(), fontsize=14)
        ax.axis("off")  # Turn off axis for clean visualization

    # Hide unused subplots
    for ax in axes.flat[len(levels_dict):]:
        ax.axis("off")

    # Add one common color bar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal',
        fraction=0.03, pad=0.1
    )
    cbar.set_label("Partworth utility", fontsize=12)

    # Save the figure
    plt.savefig(f"output/cantonal_{filename_suffix}.png", dpi=300)
    plt.show()

# %% get plots 
# plot pv experiment maps
# plot_cantonal_beta_map(levels_distribution, "distribution", vmin = None)
# plot_cantonal_beta_map(levels_tradeoffs, "tradeoffs", vmin = None)
# plot_cantonal_beta_map(levels_imports, "imports", vmin = None)

# plot heat experiment maps
plot_cantonal_beta_map(levels_year, "year", vmin = None)
plot_cantonal_beta_map(levels_tax, "tax", vmin = None)
plot_cantonal_beta_map(levels_ban, "ban", vmin = None)
plot_cantonal_beta_map(levels_exemption, "exemption", vmin = None)

# %% cantonal variance

chart = alt.Chart(cantonal_beta).encode(
    # quantitative axis for beta values
    x='beta:Q',
    # each level on separate row
    y=alt.Y('level:N', sort=desired_order), 
    # use a single color for all cantons
    color=alt.value("steelblue")
#     color='canton:N',  # Color each canton differently
).mark_circle(size=30, opacity=0.8).properties(
    width=600,
)

# Save or display the plot
chart.save("output/beta_canton_plot.html")  # Save to an HTML file
# chart.show()  # Show the plot in your notebook or IDE



# %%
