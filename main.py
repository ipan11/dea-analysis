############################
# DEA analysis and plots
############################
import subprocess
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import mlflow


mlflow.set_experiment("Risk-paper-replication")


############################
# Read input data and clean
############################
df = pd.read_csv("./datasets/DEA_data.csv")
df.drop(["Unnamed: 0"], axis=1, inplace=True)
df.columns

df.head()

############################
# Filter countries
############################

filter_list = [
    # "NRU",  # Nauru
    "AFG",  # Afghanistan
    "ALB",  # Albania
    "DZA",  # Algeria
    # "ASM",  # American Samoa
    # "AGO",  # Angola
    # "ATG",  # Antigua and Barbuda
    "ARG",  # Argentina
    "ARM",  # Armenia
    # "ABW",  # Aruba
    "AUS",  # Australia
    "AUT",  # Austria
    # "AZE",  # Azerbaijan
    # "BHS",  # Bahamas, The
    # "BHR",  # Bahrain
    "BGD",  # Bangladesh
    # "BRB",  # Barbados
    "BLR",  # Belarus
    "BEL",  # Belgium
    # "BLZ",  # Belize
    # "BEN",  # Benin
    # "BMU",  # Bermuda
    # "BTN",  # Bhutan
    "BOL",  # Bolivia
    "BIH",  # Bosnia and Herzegovina
    # "BWA",  # Botswana
    "BRA",  # Brazil
    # "BRN",  # Brunei Darussalam
    "BGR",  # Bulgaria
    # "BFA",  # Burkina Faso
    # "BDI",  # Burundi
    # "KHM",  # Cambodia
    "CMR",  # Cameroon
    "CAN",  # Canada
    # "CPV",  # Cape Verde
    # "CYM",  # Cayman Islands
    # "CAF",  # Central African Republic
    # "TCD",  # Chad
    "CHL",  # Chile
    "CHN",  # China
    "COL",  # Colombia
    # "COM",  # Comoros
    "COG",  # Congo, Rep.
    "CRI",  # Costa Rica
    "CIV",  # Cote d'Ivoire
    "HRV",  # Croatia
    "CUB",  # Cuba
    "CYP",  # Cyprus
    "CZE",  # Czech Republic
    "DNK",  # Denmark
    # "DMA",  # Dominica
    "DOM",  # Dominican Republic
    "ECU",  # Ecuador
    "EGY",  # Egypt, Arab Rep.
    "SLV",  # El Salvador
    # "GNQ",  # Equatorial Guinea
    "EST",  # Estonia
    "ETH",  # Ethiopia
    # "FJI",  # Fiji
    "FIN",  # Finland
    "FRA",  # France
    "GAB",  # Gabon
    # "GMB",  # Gambia, The
    # "GEO",  # Georgia
    "DEU",  # Germany
    "GHA",  # Ghana
    "GRC",  # Greece
    # "GRL",  # Greenland
    # "GRD",  # Grenada
    # "GUM",  # Guam
    "GTM",  # Guatemala
    # "GIN",  # Guinea
    # "GNB",  # Guinea-Bissau
    # "GUY",  # Guyana
    "HTI",  # Haiti
    # "HND",  # Honduras
    # "HKG",  # Hong Kong SAR, China
    "HUN",  # Hungary
    "ISL",  # Iceland
    "IND",  # India
    "IDN",  # Indonesia
    "IRN",  # Iran, Islamic Rep.
    "IRQ",  # Iraq
    "IRL",  # Ireland
    "ISR",  # Israel
    "ITA",  # Italy
    "JAM",  # Jamaica
    "JPN",  # Japan
    "JOR",  # Jordan
    "KAZ",  # Kazakhstan
    "KEN",  # Kenya
    # "KIR",  # Kiribati
    "KOR",  # Korea, Rep.
    # "KWT",  # Kuwait
    # "KGZ",  # Kyrgyz Republic
    # "LAO",  # Lao PDR
    # "LVA",  # Latvia
    "LBN",  # Lebanon
    # "LSO",  # Lesotho
    "LBR",  # Liberia
    # "LBY",  # Libya
    "LTU",  # Lithuania
    "LUX",  # Luxembourg
    # "MAC",  # Macao SAR, China
    "MKD",  # Macedonia, FYR
    "MDG",  # Madagascar
    # "MWI",  # Malawi
    "MYS",  # Malaysia
    # "MDV",  # Maldives
    "MLI",  # Mali
    # "MLT",  # Malta
    "MHL",  # Marshall Islands
    # "MRT",  # Mauritania
    "MUS",  # Mauritius
    "MEX",  # Mexico
    # "FSM",  # Micronesia, Fed. Sts.
    # "MDA",  # Moldova
    "MNG",  # Mongolia
    # "MNE",  # Montenegro
    "MAR",  # Morocco
    # "MOZ",  # Mozambique
    "MMR",  # Myanmar
    "NAM",  # Namibia
    # "NPL",  # Nepal
    "NLD",  # Netherlands
    "NZL",  # New Zealand
    "NIC",  # Nicaragua
    "NER",  # Niger
    "NGA",  # Nigeria
    "NOR",  # Norway
    # "OMN",  # Oman
    "PAK",  # Pakistan
    # "PLW",  # Palau
    "PAN",  # Panama
    # "PNG",  # Papua New Guinea
    "PRY",  # Paraguay
    "PER",  # Peru
    "PHL",  # Philippines
    "POL",  # Poland
    "PRT",  # Portugal
    # "PRI",  # Puerto Rico
    "QAT",  # Qatar
    "RUS",  # Russian Federation
    # "RWA",  # Rwanda
    # "WSM",  # Samoa
    # "STP",  # Sao Tome and Principe
    "SAU",  # Saudi Arabia
    "SEN",  # Senegal
    # "SRB",  # Serbia
    # "SYC",  # Seychelles
    "SLE",  # Sierra Leone
    "SGP",  # Singapore
    "SVK",  # Slovak Republic
    "SVN",  # Slovenia
    # "SLB",  # Solomon Islands
    "ZAF",  # South Africa
    # "SSD",  # South Sudan
    "ESP",  # Spain
    "LKA",  # Sri Lanka
    # "KNA",  # St. Kitts and Nevis
    # "LCA",  # St. Lucia
    # "VCT",  # St. Vincent and the Grenadines
    "SDN",  # Sudan
    # "SUR",  # Suriname
    # "SWZ",  # Swaziland
    "SWE",  # Sweden
    "CHE",  # Switzerland
    # "TJK",  # Tajikistan
    "TZA",  # Tanzania
    "THA",  # Thailand
    # "TGO",  # Togo
    # "TON",  # Tonga
    # "TTO",  # Trinidad and Tobago
    "TUN",  # Tunisia
    "TUR",  # Turkey
    # "TKM",  # Turkmenistan
    # "TUV",  # Tuvalu
    "UGA",  # Uganda
    "UKR",  # Ukraine
    "ARE",  # United Arab Emirates
    "GBR",  # United Kingdom
    "USA",  # United States
    "URY",  # Uruguay
    "UZB",  # Uzbekistan
    # "VUT",  # Vanuatu
    # "VNM",  # Vietnam
    # "VIR",  # Virgin Islands (U.S.)
    "YEM",  # Yemen, Rep.
    "ZMB",  # Zambia
    "ZWE",  # Zimbabwe
]

df = df[df["Country Code"].isin(filter_list)]
df.reset_index(drop=True, inplace=True)
print(f"Number of countries: {df.shape[0]}")

############################
# Normalise the dataset and save as csv
############################

df.dropna(inplace=True)

col_names_y = [
    "TotElecGen_2015",
    # "NotRenewShare_2015",
    "PolStab_2015",
    "ContCorrup_2015",
    "RegQual_2015",
    "RuleofLaw_2015",
    "GDP2010base_2015",
    "GDPPerCapita_2015",
    "GovEffec_2015",
]
col_names_x = [
    "RenewElecGen_2015",
]

x = df[col_names_x].to_numpy()
y = df[col_names_y].to_numpy()

# Scale each row individually
scaler = MinMaxScaler(feature_range=(0.01, 1))
# scaler.fit_transform(data[:, 0].reshape(-1, 1))
x_norm = scaler.fit_transform(x)
y_norm = scaler.fit_transform(y)
# TODO: Check vars after scaling. Renewables for example is skewed due to one country having very high value
# # # Take log of all vars to get around skew
# ## doing it by normalising and then taking log and re normalising
# ## to avoid log(0) and -ve vals for DEA input
# x_norm = scaler.fit_transform(np.log(x_norm))
# y_norm = scaler.fit_transform(np.log(y_norm))

# Normalise only selected features
x_norm = scaler.fit_transform(np.log(x_norm))
# TODO: Hard coded for now. Will be impacted if col_names_y order changes
# TotElecGen_2015
temp = scaler.fit_transform(np.log(y_norm[:, 0]).reshape(-1, 1))
y_norm[:, 0] = temp.reshape(1, -1)[0]
# GDP2010base_2015
temp = scaler.fit_transform(np.log(y_norm[:, 5]).reshape(-1, 1))
y_norm[:, 5] = temp.reshape(1, -1)[0]

# Save as csv
# TODO: refactor to normalise and save with pandas itself
pd.DataFrame(x_norm, columns=col_names_x).to_csv(
    "./datasets/x_normalised.csv", index=False
)
pd.DataFrame(y_norm, columns=col_names_y).to_csv(
    "./datasets/y_normalised.csv", index=False
)

############################
# Run DEA pipeline R script
############################
subprocess.run("Rscript runDEA.R", shell=True)

############################
# Post process values and plot
############################
results = pd.read_csv("./results/dea_result.csv")
results.head()
results.columns
# Set country names as index
results["Country Name"] = df["Country Name"]
results.drop(["Unnamed: 0"], axis=1, inplace=True)
# Columns are input_x, input_y, efficiency, weight_x, weight_y

# Plot overall efficiencis to compare with python model
# fig = px.bar(results, x="Country Name", y="Eff")
# fig.show()


### Debugging model inputs
df_norm_data = pd.DataFrame(y_norm)
df_norm_data[df_norm_data.shape[1] + 1] = x_norm

df_norm_data.columns = sum([col_names_y, col_names_x], [])  # flatten list

df_norm_data.index = results["Country Name"]  # set index as country names
fig = px.imshow(
    df_norm_data.T, aspect="auto", title="Normalised data inputs into the DEA model"
)
# fig.show()
plotly.offline.plot(fig, filename="./plots/normed_data.html")

#################################
# Plot efficiency result on a world map

results["Country Code"] = df["Country Code"]
fig = px.choropleth(
    results,
    locations="Country Code",
    color="Eff",  # lifeExp is a column of gapminder
    hover_name="Country Name",  # column to add to hover information
    color_continuous_scale=px.colors.sequential.Magma,
)
# fig.show()
plotly.offline.plot(fig, filename="./plots/total_efficiency_world_map.html")

#################################
# Compute two efficiencies by multiplying weights with inputs and outputs
# and plot on 2 axis

# Assumption: Economic risk is weights*output_vars and normalised with the lowest value


# Calculate economic risk
econ_risk_col_names_y = [
    "GDP2010base_2015",
    "GDPPerCapita_2015",
    "GovEffec_2015",
]

numerator = (results[econ_risk_col_names_y].to_numpy()) * (
    results[list(map(lambda x: x + ".1", econ_risk_col_names_y))].to_numpy()
)  # normalised_output*output_weights

econ_risk_eff = numerator.sum(axis=1)

df_econ_risk_eff = pd.DataFrame(econ_risk_eff, columns=["Econ_risk_eff"])
df_econ_risk_eff["Country Code"] = results["Country Code"]
df_econ_risk_eff["Country Name"] = results["Country Name"]

# TODO: Have to define relative risk here, but division by zero in current form

# Calculate institutional risk
inst_risk_col_names_y = [
    "PolStab_2015",
    "ContCorrup_2015",
    "RegQual_2015",
    "RuleofLaw_2015",
]

numerator = (results[inst_risk_col_names_y].to_numpy()) * (
    results[list(map(lambda x: x + ".1", inst_risk_col_names_y))].to_numpy()
)  # normalised_output*output_weights

inst_risk_eff = numerator.sum(axis=1)

df_inst_risk_eff = pd.DataFrame(inst_risk_eff, columns=["Inst_risk_eff"])
df_inst_risk_eff["Country Code"] = results["Country Code"]
df_inst_risk_eff["Country Name"] = results["Country Name"]


df_all_risks = df_inst_risk_eff.copy()
df_all_risks["Econ_risk_eff"] = df_econ_risk_eff["Econ_risk_eff"]

fig = px.scatter(
    df_all_risks,
    x="Inst_risk_eff",
    y="Econ_risk_eff",
    text="Country Name",
    hover_name="Country Code",
    # hover_data=["continent", "pop"],
)

# fig.show()
plotly.offline.plot(fig, filename="./plots/2axis_risk.html")


fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        "Institutional Risk factors efficiency",
        "Economic Risk factors efficiency",
    ),
)

fig.add_trace(
    go.Bar(
        y=df_all_risks["Econ_risk_eff"].values.T,
        x=df_all_risks["Country Name"].values.T,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(
        y=df_all_risks["Inst_risk_eff"].values.T,
        x=df_all_risks["Country Name"].values.T,
    ),
    row=2,
    col=1,
)
# fig.show()
plotly.offline.plot(fig, filename="./plots/Insti_Econ_risk_bar.html")

# fig = px.bar(df_all_risks, x="Country Name", y="Econ_risk_eff")
# fig.show()

# fig = px.bar(df_all_risks, x="Country Name", y="Inst_risk_eff")
# fig.show()
#################################
# Plot heatmap of input*weight, output*weight, efficiency together

# Weighted y
col_weights_y = list(map(lambda x: x + ".1", col_names_y))
col_names_y

weighted_y = pd.DataFrame()

for i, j in zip(col_weights_y, col_names_y):
    print(i, j)
    weighted_y[f"{j}_weighted"] = results[i] * results[j]

weighted_y.index = results["Country Name"]
fig = px.imshow(weighted_y.T, aspect="auto")
fig.update_layout(title="Outputs multiplied by weights (numerator)")
# fig.show()
plotly.offline.plot(fig, filename="./plots/Outputs_multiplied_by_weights.html")

# Weighted x
col_weights_x = list(map(lambda x: x + ".1", col_names_x))
col_names_x

weighted_x = pd.DataFrame()

for i, j in zip(col_weights_x, col_names_x):
    print(i, j)
    weighted_x[f"{j}_weighted"] = results[i] * results[j]

weighted_x.index = results["Country Name"]
fig = px.imshow(weighted_x.T, aspect="auto")
fig.update_layout(title="Inputs multiplied by weights (denominator)")
# fig.show()
plotly.offline.plot(fig, filename="./plots/Inputs_multiplied_by_weights.html")
# Efficiency dataframe along with weighted inputs and outputs

fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=("Weighted & summed y", "Weighted & summed x", "Efficiency"),
)

fig.add_trace(
    go.Bar(y=weighted_y.sum(axis=1).values.T, x=weighted_y.sum(axis=1).index.T),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(y=weighted_x.sum(axis=1).values.T, x=weighted_x.sum(axis=1).index.T),
    row=2,
    col=1,
)

fig.add_trace(
    go.Bar(y=results["Eff"].values.T, x=results["Country Name"].values.T), row=3, col=1
)

# fig.show()
plotly.offline.plot(fig, filename="./plots/Efficiency_summed_inputs_outputs.html")

#################################
## Log code and artifacts to mlflow
if not os.path.exists("artifacts"):
    os.makedirs("artifacts")

# Copy folder - results, plots, main.py and runDEA.R into artifacts and log

shutil.copytree("./plots", "./artifacts/plots")
shutil.copytree("./results", "./artifacts/results")
# shutil.copyfile("./data_scraper.py", "./artifacts/data_scraper.py")
shutil.copyfile("./main.py", "./artifacts/main.py")
shutil.copyfile("./runDEA.R", "./artifacts/runDEA.R")

mlflow.log_artifacts("artifacts")

# Log the input and output variable names
mlflow.log_param("col_names_x", col_names_x)
mlflow.log_param("col_names_y", col_names_y)
