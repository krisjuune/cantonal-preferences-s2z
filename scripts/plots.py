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

desired_order = desired_order_pv

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

beta_0 = cantonal_beta[cantonal_beta['level'] == "imports_0%"]
beta_10 = cantonal_beta[cantonal_beta['level'] == "imports_10%"]
beta_20 = cantonal_beta[cantonal_beta['level'] == "imports_20%"]
beta_30 = cantonal_beta[cantonal_beta['level'] == "imports_30%"]

# heat
beta_phaseout = cantonal_beta[cantonal_beta['level'] == "year_2040"]
beta_tax = cantonal_beta[cantonal_beta['level'] == "tax_100%"]
beta_subsidy = cantonal_beta[cantonal_beta['level'] == "heatpump_subsidy"]
beta_lowincome = cantonal_beta[cantonal_beta['level'] == "exemption_low"]

# Define levels and data for each map

# pv
levels_pv = {
    "distribution_potential-based": beta_potential,
    "imports_0%": beta_0,
    "tradeoffs_alpine": beta_alpine,
    "tradeoffs_lakes": beta_lakes,
}

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
    "imports_0%": beta_0, 
    "imports_10%": beta_10, 
    "imports_20%": beta_20, 
    "imports_30%": beta_30
}

# heat
levels_heat = {
    "year_2040": beta_phaseout,
    "tax_100%": beta_tax,
    "heatpump_subsidy": beta_subsidy,
    "exemption_low": beta_lowincome
}

# levels = levels_tradeoffs
# levels = levels_distribution
levels = levels_imports

# %% maps that don't work

# choose level for plotting
beta_per_level = beta_alpine

# Merge the beta data with the GeoDataFrame by matching canton names
merged_df = cantons.merge(beta_per_level, left_on="NAME", right_on="canton", how="left")
merged_df = merged_df.drop(columns=["DATUM_AEND", "DATUM_ERST"])

# Convert the merged GeoDataFrame to a GeoJSON format for Altair
merged_geojson = json.loads(merged_df.to_json())

# Create an Altair chart using the GeoJSON data
chart = alt.Chart(alt.Data(values=merged_geojson)).mark_geoshape().encode(
    color=alt.Color('beta:Q', scale=alt.Scale(scheme='redblue', domainMid=0), title="Partworth"),
    tooltip=['NAME:N', 'beta:Q']  # Optional: show canton name and beta value on hover
).properties(
    width=500,
    height=500,
    title="Distribution Potential-based Beta Values across Swiss Cantons"
)

# Save or display the map
chart.save("map_distribution_potential_pv.html")

plot with matplotlib
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged_df.plot(column='beta', cmap='viridis', legend=True, ax=ax)
#coolmap for blue to red

# Add title and display
plt.title("Support for renewable energy infrastructure in Alpine regions")
plt.show()

# %% plotting attribute levels on map

# Set up a color map and normalization to standardize color scale across maps
# cmap = plt.cm.viridis
cmap = plt.cm.coolwarm.reversed()

norm = mcolors.Normalize(vmin=-0.4, vmax=beta_0['beta'].max())
# norm = mcolors.Normalize(vmin=cantonal_beta['beta'].min(), vmax=cantonal_beta['beta'].max())

fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

# Iterate over levels to plot each map in a subplot
for ax, (title, beta_level) in zip(axes.flat, levels.items()):
    # Merge data for the specific level
    beta_per_level = beta_level
    merged_df = cantons.merge(beta_per_level, left_on="NAME", right_on="canton", how="left")
    
    # Plot each level with shared color scale
    merged_df.plot(column='beta', cmap=cmap, legend=False, ax=ax, norm=norm)
    ax.set_title(f"{title.replace('_', ' ').capitalize()}")

# Add one common color bar on the right of the figure
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal',
    fraction=0.03, pad=0.1
)
cbar.set_label("Partworth utility", fontsize=12)

plt.show()



# %%
