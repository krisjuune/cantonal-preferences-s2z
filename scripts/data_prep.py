import pandas as pd
import numpy as np 
from functions.data_assist import apply_mapping, rename_columns
from functions.conjoint_assist import prep_conjoint


#%% ############################# read data ##################################

df = pd.read_csv('raw_data/raw_conjoint_120624.csv', low_memory = False, skiprows = [1,2])

# check data 
pd.set_option('display.max_columns', None)
columns = df.columns.tolist()



# %% ############################# clean data ################################

# fix typos and replace dashes with underscores
df.rename(columns={'languge': 'language'}, inplace=True)
df = rename_columns(df, 'justice-', 'justice_')

# fix data types
df['Finished'] = df['Finished'].replace(
    {'true': True, 'True': True, 'false': False, 'False': False}
).astype(bool)

columns_to_num = ['Duration (in seconds)', 
                  'household-size', 
                  'trust_1', 
                  'trust_2',
                  'trust_3', 
                  'satisfaction_1', 
                  'literacy6_5']
for col in columns_to_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# add column for IDs and duration in min
df['ID'] = range(1, len(df) + 1)
df['duration_min'] = df['Duration (in seconds)'] / 60
df['duration_min'].round(3) # do I need to store it in df['dur...'] as well?

# filter out previews
df = df[df['DistributionChannel'] != 'preview'] 
# filter out recorded incompletes
df = df[df['Finished'] == True] 
# filter out quota fulls
df = df.dropna(subset=['canton']) 

# speeders and laggards
# calculate the 5% and 99% quantiles
lower_threshold = df['duration_min'].quantile(0.05)
upper_threshold = df['duration_min'].quantile(0.95)
print(f"Lower threshold (lowest 5% quartile): {lower_threshold} minutes")
print(f"Upper threshold (highest 5% quartile): {upper_threshold} minutes")
df['speeder'] = df['duration_min'] < lower_threshold
df['laggard'] = df['duration_min'] > upper_threshold 

# inattentives based on justice section (exact same answer for all questions)
just_columns = ['justice_general_1', 'justice_tax_1', 'justice_subsidy_1', 
           'justice_general_2', 'justice_tax_2', 'justice_subsidy_2', 
           'justice_general_3', 'justice_tax_3', 'justice_subsidy_3', 
           'justice_general_4', 'justice_tax_4', 'justice_subsidy_4'
]
attention_mask = (df[just_columns].nunique(axis=1) == 1)
df['inattentive'] = attention_mask

# count the number of rows where the attention filters are True
print(f"Number of speeders (5% fastest): {df['speeder'].sum()}")
print(f"Number of laggards (5% slowest): {df['laggard'].sum()}")
print(f"Number of inattentive respondents: {df['inattentive'].sum()}")

# filter out rows of speeders, laggards, or inattentives
df_filtered = df[~((df['speeder'] == True) | 
                   (df['laggard'] == True) |
                   (df['inattentive'] == True)
                  )]

df = df_filtered

# remove non-functional empty columns 
empty_columns = [col for col in df.columns if col.endswith('_Table')]
df = df.drop(columns=empty_columns)

# rename columns for pv experiment
df = rename_columns(df, 'TargetMix', 'mix')
df = rename_columns(df, 'Imports', 'imports')
df = rename_columns(df, 'RooftopSolarPV', 'pv')
df = rename_columns(df, 'Infrastructure', 'tradeoffs')
df = rename_columns(df, 'Distribution', 'distribution')



# %% ########################## recode demographics ###########################

# recode likert scales in conjoints
numerical_values = [0, 1, 2, 3, 4, 5]
rating_values = ['Stark dagegen',
                 'Dagegen',
                 'Eher dagegen',
                 'Eher dafür',
                 'Dafür',
                 'Stark dafür']
rating_scale = np.array(list(zip(rating_values, numerical_values)))
likert_dict = {**dict(rating_scale)}
df = apply_mapping(df, likert_dict, column_pattern=['justice', 'rating'])

# recode demographic values

demographics_dict = {
    # gender
    "Weiblich": "female",
    "Männlich": "male", 
    "Nicht-binär": "non-binary",

    # age
    "18-39 Jahre": "18-39", 
    "40-64 Jahre": "40-64", 
    "65-79 Jahre": "65-79", 
    "80 Jahre oder älter": "80+",

    # language region
    "Deutschsprachige Schweiz": "german", 
    "Französischsprachige Schweiz": "french", 
    "Italienischsprachige Schweiz": "italian",
    "Rätoromanische Schweiz": "romansh",

    # survey language
    "Deutsch": "german",
    "Französisch": "french",
    "Italienisch": "italian",

    # income
    "Unter CHF 70,000": "low", # lower
    "CHF 70,000 – CHF 100,000": "mid", # lower
    "CHF 100,001 – CHF 150,000": "mid", # higher
    "CHF 150,001 – CHF 250,000": "high", # higher
    "Über 250,000": "high", # higher
    "Möchte ich nicht sagen": np.nan, 

    # education 
    "Keine Matura": "no secondary",
    "Matura oder Berufsausbildung": "secondary",
    "Abschluss einer Fachhochschule oder Universität": "university",

    # citizen 
    "Ja": True, 
    "Nein": False, 

    # tenant
    "Mieter:in": True, 
    "Besitzer:in": False,

    # urbanness
    "Stadt": "city",
    "Agglomeration": "suburb",
    "Land": "rural",

    # political orientation
    "Grüne Partei der Schweiz (GPS)": "left", 
    "Sozialdemokratische Partei der Schweiz (SP)": "left", 
    "Grünliberale Partei (GLP)": "liberal", 
    "Die Mitte (ehemals CVP/BDP)": "liberal", 
    "Die Liberalen (FDP)": "conservative", 
    "Schweizerische Volkspartei (SVP)": "conservative", 
    "Andere": np.nan, 
    "Keine": np.nan, 
    "Möchte ich nicht sagen": np.nan,

    # canton to match Swiss Boundaries 3D data 
    "Geneva": "Genève",
    "Lucerne": "Luzern",
    "Zurich": "Zürich"

    #TODO energy literacy
}

# apply mapping to columns whose names contain 'table'
df = apply_mapping(df, demographics_dict)

#TODO household size ?

# create categorical political trust and governmental satisfaction
df = df.copy() # reduce fragmentation
df['trust_mean'] = pd.concat([df['trust_1'], df['trust_2'], df['trust_3']], axis=1).mean(axis=1).round(3)
df['trust'] = pd.cut(df['trust_mean'], 
                              bins=[-float('inf'), 
                                    df['trust_mean'].quantile(0.33), 
                                    df['trust_mean'].quantile(0.65), # ensures ~33% in each bin
                                    float('inf')], 
                              labels=['low', 'mid', 'high'], 
                              include_lowest=True)

df['satisfaction'] = pd.cut(df['satisfaction_1'], 
                              bins=[-float('inf'), 
                                    df['satisfaction_1'].quantile(0.27), 
                                    df['satisfaction_1'].quantile(0.63), # ensures ~33% in each bin
                                    float('inf')], 
                              labels=['low', 'mid', 'high'], 
                              include_lowest=True)

# %% ########################## translate conjoints ###########################

translation_dict_heat = {
    # ban
    "Kein Verbot": "No ban",
    "Pas d'interdiction": "No ban",
    "Nessun divieto": "No ban",

    "Verbot von Neuinstallationen": "Ban on new installations",
    "Interdiction de nouvelles installations uniquement": "Ban on new installations",
    "Divieto di installare nuovi boiler": "Ban on new installations",

    "Verbot von Neuinstallationen und obligatorischer Austausch bestehender fossilen Heizungen": "Ban and fossil heating replacement",
    "Interdiction de nouvelles installations et remplacement obligatoire des chauffages à combustibles fossiles existants": "Ban and fossil heating replacement",
    "Divieto di installare nuovi boiler e sostituzione obbligatoria dei boiler esistenti": "Ban and fossil heating replacement",

    # heat pump
    "Wärmepumpe mit Subventionen kaufen": "Subsidy", 
    "Achat d’une pompe à chaleur avec des subventions": "Subsidy",
    "Acquisto di una pompa di calore con sovvenzioni": "Subsidy",

    "Wärmepumpe von der Regierung leasen": "Governmental lease",
    "Achat d’une pompe à chaleur en leasing auprès du gouvernement": "Governmental lease",
    "Leasing di una pompa di calore di proprietà del governo": "Governmental lease",

    "Wärmepumpen-Abo": "Subscription",
    "Abonnement à une pompe à chaleur": "Subscription",
    "Abbonamento ad una pompa di calore": "Subscription",

    # building codes
    "Neue Gebäude müssen energieeffizient sein": "New buildings must be energy efficient", 
    "Les nouveaux bâtiments doivent être énergétiquement efficaces": "New buildings must be energy efficient",
    "Nuovi edifici devono rispettare standard di alta efficienza energetica": "New buildings must be energy efficient",

    "Neue Gebäude müssen energieeffizient sein und vor Ort erneuerbaren Strom erzeugen": "New buildings must be energy efficient and produce renewable electricity on-site",
    "Les nouveaux bâtiments doivent être énergétiquement efficaces et produire de l'électricité renouvelable sur place": "New buildings must be energy efficient and produce renewable electricity on-site",
    "Nuovi edifici devono rispettare standard di alta efficienza energetica e produrre elettricità rinnovabile in modo autonomo": "New buildings must be energy efficient and produce renewable electricity on-site",

    "Alle Gebäude müssen energieeffizient sein": "All buildings need to be energy efficient",
    "Tous les bâtiments doivent être énergétiquement efficaces": "All buildings need to be energy efficient",
    "Tutti gli edifici devono rispettare standard di alta efficienza energetica": "All buildings need to be energy efficient",

    "Alle Gebäude müssen energieeffizient sein und vor Ort erneuerbaren Strom erzeugen": "All buildings need to be energy efficient and produce renewable electricity on-site",
    "Tous les bâtiments doivent être énergétiquement efficaces et produire de l'électricité renouvelable sur place": "All buildings need to be energy efficient and produce renewable electricity on-site",
    "Tutti gli edifici devono rispettare standard di alta efficienza energetica e produrre elettricità rinnovabile in modo autonomo": "All buildings need to be energy efficient and produce renewable electricity on-site",
    
    # exemptions -- there's an error here somewhere
    "Keine Ausnahmen": "No exemptions", 
    "Pas d'exemption": "No exemptions",
    "Nessuna eccezione": "No exemptions",

    "Geringverdienende Haushalte sind ausgenommen": "Low-income households are exempted",
    "Les ménages à revenus faibles sont exclus": "Low-income households are exempted",
    "Sono esentate le famiglie e utenze a basso reddito": "Low-income households are exempted",

    "Gering- und mittelverdienende Haushalte sind ausgenommen": "Low and middle-income households are exempted",
    "Les ménages à revenus faibles et moyens sont exclus": "Low and middle-income households are exempted",
    "Sono esentate le famiglie e utenze a basso e medio reddito": "Low and middle-income households are exempted"
}

translate_dict_pv = {
    # target mix
    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_Xuqo08nWGvzTaSr': 'More hydro',
    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_lwjCDBh17ODzYQM': 'More hydro', 
    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_FvSefnnxSgWbb8J': 'More hydro', 

    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_PnFZWmknO1NZLvB': 'More solar', 
    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_vCbbVKg7jmWJgva': 'More solar', 
    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_WFCdHR97e3KUwQG': 'More solar', 

    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_9LCSI0Qu1yQuHNY': 'More wind',
    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_9dSwpo1C4dEgjHD': 'More wind', 
    'https://climatepolicy.qualtrics.com/ControlPanel/Graphic.php?IM=IM_G9HNH3uNGMuVtEb': 'More wind', 

    # rooftop pv requirements
    'Keine Verpflichtungen': 'No obligation', 
    'Nessun obbligo': 'No obligation', 
    'Aucune obligation': 'No obligation', 

    'Neuen öffentlichen und gewerblichen Gebäuden': 'New public and commercial buildings', 
    'Les nouveaux bâtiments publics et commerciaux': 'New public and commercial buildings', 
    'Nuovi edifici pubblici e commerciali': 'New public and commercial buildings',

    'Neuen und existierenden öffentlichen und gewerblichen Gebäuden': 'New and existing public and commercial buildings', 
    'Les bâtiments publics et commerciaux à la fois nouveaux et existants': 'New and existing public and commercial buildings', 
    'Edifici pubblici e commerciali sia nuovi che esistenti': 'New and existing public and commercial buildings', 

    'Allen neuen Gebäuden': 'All new buildings', 
    'Tous les nouveaux bâtiments': 'All new buildings', 
    'Tutti i nuovi edifici': 'All new buildings', 

    'Allen neuen und existierenden Gebäuden': 'All new and existing buildings', 
    'Tous les bâtiments neufs et existants': 'All new and existing buildings', 
    'Tutti gli edifici nuovi ed esistenti': 'All new and existing buildings', 

    # biodiversity tradeoffs
    'Keine Ausnahmefälle': 'No trade-offs',
    'Pas de cas exceptionnels': 'No trade-offs', 
    'In nessun caso eccezionale': 'No trade-offs', 

    'Alpenregionen': 'Alpine regions',
    'Les régions alpines': 'Alpine regions', 
    'Regioni alpine': 'Alpine regions', 

    'Landwirtschaflichen Flächen': 'Agricultural areas',
    'Les terres agricoles': 'Agricultural areas', 
    'Superfici agricole': 'Agricultural areas',

    'Wäldern': 'Forests',
    'Les forêts': 'Forests', 
    'Foreste': 'Forests', 

    'Flüssen': 'Rivers',
    'Les rivières': 'Rivers', 
    'Fiumi': 'Rivers',

    'Seen': 'Lakes', 
    'Les lacs': 'Lakes', 
    'Laghi': 'Lakes',

    # cantonal distribution
    'Keine Vorgabe': 'No agreed distribution', 
    'Pas d\'objectif': 'No agreed distribution', 
    'Nessun obiettivo': 'No agreed distribution', 

    'Basierend auf dem Erzeugungspotenzial': 'Potential-based', 
    'Basée sur la production maximale potentielle d’un canton': 'Potential-based', 
    'In base al potenziale di un cantone': 'Potential-based', 

    'Basierend auf der Bevölkerungszahl': 'Equal per person', 
    'Basée sur le nombre de personnes vivant dans chaque canton': 'Equal per person',
    'In base al numero di abitanti di ogni cantone': 'Equal per person', 

    'Mindestensvorgabe pro Kanton': 'Minimum limit', 
    'Un minimum de production par canton est établi': 'Minimum limit', 
    'In base al livello di produzione minimo cantonale concordato': 'Minimum limit', 

    'Deckelung pro Kanton': 'Maximum limit',
    'Un maximum de production par canton est établi': 'Maximum limit',
    'Nessun cantone produce più di un tetto massimo concordato': 'Maximum limit'
}

# apply mapping to columns whose names contain 'table'
conjoint_dict = translation_dict_heat | translate_dict_pv
df = apply_mapping(df, conjoint_dict, column_pattern='table')



# %% ########################## prep conjoint data ############################

# select respondent data
respondents = df[[
        "ID", "duration_min", "gender", "age", "region", "canton", "citizen", 
        "education", "urbanness", "renting", "income", "household-size", "party", 
        "satisfaction", "speeder", "laggard", "inattentive", "trust"]] 

#TODO add energy literacy and justice

heat_regex = 'pv|mix|imports|tradeoffs|distribution'
heat_filemarker = 'heat'
pv_regex = 'heat|year|tax|ban|energyclass|exemption'
pv_filemarker = 'pv'

df_heat = prep_conjoint(df, respondent_columns=respondents, regex_list=heat_regex, filemarker=heat_filemarker) 
df_pv = prep_conjoint(df, respondent_columns=respondents, regex_list=pv_regex, filemarker=pv_filemarker) 


# %%
