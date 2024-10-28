library(broom)
library(dplyr)
library(survey)
library(marginaleffects)
library(ggplot2)
library(gridExtra)

filter_respondents <- function(df, 
                               filter_speeders = TRUE, 
                               filter_laggards = TRUE, 
                               filter_inattentives = TRUE) {
  # get initial nr of respondents
  initial_unique_ids <- df %>% pull(ID) %>% unique() %>% length()
  
  # convert the columns to logical type
  df <- df %>%
    mutate(
      speeder = as.logical(speeder),
      laggard = as.logical(laggard),
      inattentive = as.logical(inattentive)
    )
  
  # apply filtering as requested
  if (filter_speeders) {
    df <- df %>% filter(!speeder)
  }
  if (filter_laggards) {
    df <- df %>% filter(!laggard)
  }
  if (filter_inattentives) {
    df <- df %>% filter(!inattentive)
  }
  
  # get number of respondents filtered out
  final_unique_ids <- df %>% pull(ID) %>% unique() %>% length()
  filtered_out_count <- initial_unique_ids - final_unique_ids
  cat("Number of unique respondents (IDs) filtered out:", filtered_out_count, "\n")
  
  return(df)
}

theme_nice <- function() {
  theme_minimal(base_family = "Jost-Regular") +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(family = "Jost-Bold",
                                    face = "bold",
                                    hjust = 0.5),
          axis.title = element_text(family = "Jost-Medium"),
          axis.title.x = element_text(hjust = 0.5),
          axis.title.y = element_text(hjust = 1),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          strip.text = element_text(family = "Jost-Bold", face = "bold",
                                    size = rel(0.75), hjust = 0),
          strip.background = element_rect(fill = "grey90", color = NA))
}

factor_conjoint <- function(df, experiment) {
  ### check and factorise justice_class
  if ("justice_class" %in% colnames(df)) {
    if (!is.numeric(df$justice_class)) {
      df <- df %>%
        mutate(justice_class = as.numeric(justice_class))
    }
    df <- df %>%
      mutate(
        justice_class = factor(
          case_when(
            justice_class == 1 ~ 0,
            justice_class == 2 ~ 1,
            justice_class == 3 ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("egalitarian", "utilitarian", "universal")
        )
      )
  }
  
  ### check and factorise other demographic columns
  if ("gender" %in% colnames(df)) {
    df <- df %>%
      mutate(
        gender = factor(
          case_when(
            gender == "female" ~ 0,
            gender == "male" ~ 1,
            gender == "non-binary" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("female", "male", "non-binary")
        )
      )
  }

  if ("age" %in% colnames(df)) {
    df <- df %>%
      mutate(
        age = factor(
          case_when(
            age == "18-39" ~ 0,
            age == "40-64" ~ 1,
            age == "65-79" ~ 2,
            age == "80+" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("18-39", "40-64", "65+")
        )
      )
  }

  if ("region" %in% colnames(df)) {
    df <- df %>%
      mutate(
        region = factor(
          case_when(
            region == "german" ~ 0,
            region == "french" ~ 1,
            region == "italian" ~ 2,
            region == "romansh" ~ 3,
            TRUE ~ NA_real_
          ),
          levels = 0:3,
          labels = c("german", "french", "italian", "romansh")
        )
      )
  }

  if ("language" %in% colnames(df)) {
    df <- df %>%
      mutate(
        language = factor(
          case_when(
            language == "german" ~ 0,
            language == "french" ~ 1,
            language == "italian" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("german", "french", "italian")
        )
      )
  }

  if ("education" %in% colnames(df)) {
    df <- df %>%
      mutate(
        education = factor(
          case_when(
            education == "no secondary" ~ 0,
            education == "secondary" ~ 1,
            education == "university" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("no secondary", "secondary", "university")
        )
      )
  }
  
  if ("income" %in% colnames(df)) {
    df <- df %>%
      mutate(
        income = factor(
          case_when(
            income == "low" ~ 0,
            income == "mid" ~ 1,
            income == "high" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("low", "mid", "high")
        )
      )
  }
  
  if ("citizen" %in% colnames(df)) {
    df <- df %>%
      mutate(
        citizen = factor(
          case_when(
            citizen == TRUE ~ 0,
            citizen == FALSE ~ 1,
            TRUE ~ NA_real_
          ),
          levels = 0:1,
          labels = c("yes", "no")
        )
      )
  }
  
  if ("renting" %in% colnames(df)) {
    df <- df %>%
      mutate(
        citizen = factor(
          case_when(
            renting == TRUE ~ 0,
            renting == FALSE ~ 1,
            TRUE ~ NA_real_
          ),
          levels = 0:1,
          labels = c("yes", "no")
        )
      )
  }

  if ("party" %in% colnames(df)) {
    df <- df %>%
      mutate(
        party = factor(
          case_when(
            party == "left" ~ 0,
            party == "liberal" ~ 1,
            party == "conservative" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("left", "liberal", "conservative")
        )
      )
  }
  
  if ("urbanness" %in% colnames(df)) {
    df <- df %>%
      mutate(
        urbanness = factor(
          case_when(
            urbanness == "city" ~ 0,
            urbanness == "suburb" ~ 1,
            urbanness == "rural" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("city", "suburb", "rural")
        )
      )
  }

  if ("trust" %in% colnames(df)) {
    df <- df %>%
      mutate(
        trust = factor(
          case_when(
            trust == "low" ~ 0,
            trust == "mid" ~ 1,
            trust == "high" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("low", "mid", "high")
        )
      )
  }

    if ("satisfaction" %in% colnames(df)) {
    df <- df %>%
      mutate(
        satisfaction = factor(
          case_when(
            satisfaction == "low" ~ 0,
            satisfaction == "mid" ~ 1,
            satisfaction == "high" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("low", "mid", "high")
        )
      )
  }

  ### Factorise conjoints
  if (experiment == "heat") {
    df <- df %>%
      mutate(
        year = factor(
          case_when(
            year == "2050" ~ 0,
            year == "2045" ~ 1,
            year == "2040" ~ 2,
            year == "2035" ~ 3,
            year == "2030" ~ 4,
            TRUE ~ NA_real_
          ),
          levels = 0:4,
          labels = c("2050", "2045", "2040", "2035", "2030")
        ),
        tax = factor(
          case_when(
            tax == "0%" ~ 0,
            tax == "25%" ~ 1,
            tax == "50%" ~ 2,
            tax == "75%" ~ 3,
            tax == "100%" ~ 4,
            TRUE ~ NA_real_
          ),
          levels = 0:4,
          labels = c("0%", "25%", "50%", "75%", "100%")
        ),
        ban = factor(
          case_when(
            ban == "No ban" ~ 0,
            ban == "Ban on new installations" ~ 1,
            ban == "Ban and fossil heating replacement" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("No ban",
                     "Ban new installations",
                     "Ban and replace fossil heating")
        ),
        heatpump = factor(
          case_when(
            heatpump == "Subsidy" ~ 0,
            heatpump == "Governmental lease" ~ 1,
            heatpump == "Subscription" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("Subsidized heat pump",
                     "Leased heat pump",
                     "Heat pump subscription")
        ),
        energyclass = factor(
          case_when(
            energyclass == "New buildings must be energy efficient" ~ 0,
            energyclass == "New buildings must be energy efficient and produce renewable electricity on-site" ~ 1,
            energyclass == "All buildings need to be energy efficient" ~ 2,
            energyclass == "All buildings need to be energy efficient and produce renewable electricity on-site" ~ 3,
            TRUE ~ NA_real_
          ),
          levels = 0:3,
          labels = c("New buildings efficient",
                     "New buildings efficient and renewable",
                     "All buildings efficient",
                     "All buildings efficient and renewable")
        ),
        exemption = factor(
          case_when(
            exemption == "No exemptions" ~ 0,
            exemption == "Low-income households are exempted" ~ 1,
            exemption == "Low and middle-income households are exempted" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("No exemptions",
                     "Low-income exempted",
                     "Low- and middle-income exempted")
        )
      )
  } else if (experiment == "pv") {
    df <- df %>%
      mutate(
        mix = factor(
          case_when(
            mix == "More hydro" ~ 0,
            mix == "More solar" ~ 1,
            mix == "More wind" ~ 2,
            TRUE ~ NA_real_
          ),
          levels = 0:2,
          labels = c("More hydro", "More solar", "More wind")
        ),
        imports = factor(
          case_when(
            imports == "0%" ~ 0,
            imports == "10%" ~ 1,
            imports == "20%" ~ 2,
            imports == "30%" ~ 3,
            TRUE ~ NA_real_
          ),
          levels = 0:3,
          labels = c("0%", "10%", "20%", "30%")
        ),
        pv = factor(
          case_when(
            pv == "No obligation" ~ 0,
            pv == "New public and commercial buildings" ~ 1,
            pv == "New and existing public and commercial buildings" ~ 2,
            pv == "All new buildings" ~ 3,
            pv == "All new and existing buildings" ~ 4,
            TRUE ~ NA_real_
          ),
          levels = 0:4,
          labels = c("No rooftop PV obligation",
                     "New non-residential buildings",
                     "New and existing non-residential buildings",
                     "All new buildings",
                     "All new and existing buildings")
        ),
        tradeoffs = factor(
          case_when(
            tradeoffs == "No trade-offs" ~ 0,
            tradeoffs == "Alpine regions" ~ 1,
            tradeoffs == "Agricultural areas" ~ 2,
            tradeoffs == "Forests" ~ 3,
            tradeoffs == "Rivers" ~ 4,
            tradeoffs == "Lakes" ~ 5,
            TRUE ~ NA_real_
          ),
          levels = 0:5,
          labels = c("No biodiversity trade-offs",
                     "Alpine regions",
                     "Agricultural regions",
                     "Forests",
                     "Rivers",
                     "Lakes")
        ),
        distribution = factor(
          case_when(
            distribution == "No agreed distribution" ~ 0,
            distribution == "Potential-based" ~ 1,
            distribution == "Equal per person" ~ 2,
            distribution == "Minimum limit" ~ 3,
            distribution == "Maximum limit" ~ 4,
            TRUE ~ NA_real_
          ),
          levels = 0:4,
          labels = c("No agreed cantonal production requirements",
                     "Maximum production potential",
                     "Equal per person",
                     "Minimum limit",
                     "Maximum limit")
        )
      )
  } else {
    stop("Error: experiment must be either 'pv' or 'heat'.")
  }
  
  return(df)
}


marginal_means <- function(df, 
                           response_var, 
                           predictors, 
                           output_file = "data/MMs.csv", 
                           id_column = "ID") 
  {
  
  # create the formula using independent and dependent vars
  formula <- as.formula(paste(response_var, "~", paste(predictors, collapse = " + ")))
  
  # Linear model with survey design
  lin_model_pv <- lm(
    formula,
    data = df
  )
  
  svydesign_pv <- svydesign(
    ids = df[[id_column]],
    weights = 1,  # Add proper weights if needed
    data = df
  )
  
  model_pv <- svyglm(
    formula,
    design = svydesign_pv
  )
  
  # Compute marginal effects
  mfx_pv <- model_pv %>%
    avg_slopes(newdata = "mean") 
  
  #TODO the plotting is messy still, the order of attributes and levels 
  #gets messed up
  
  # Step 5: Prepare Data for Plotting
  plot_data <- model_pv %>%
    tidy_and_attach() %>%
    tidy_add_reference_rows() %>%
    tidy_add_estimate_to_reference_rows() %>%
    filter(term != "(Intercept)")
  
  # Step 6: Plot
  p <- ggplot(plot_data, aes(x = estimate, y = term)) +
    geom_vline(xintercept = 0) +
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high)) +
    labs(
      title = paste("Model for", response_var),
      x = "Estimate",
      y = "Terms"
    ) +
    theme_minimal()
  
  # Print the plot
  print(p)
  
  # Save the data required for plotting
  write.csv(plot_data, file = output_file, row.names = FALSE)
  print(paste("MMs data for plotting saved to", output_file))
  
  return(list(lin_model_summary = tidy(lin_model), svyglm_model = model_pv, marginal_effects = mfx_pv, plot = p))
}

subgroup_mm <- function(df, 
                        by,
                        experiment="heat", 
                        save_file=TRUE, 
                        get_plot=FALSE)
  {
  
  # get predictors per experiment
  if (experiment == "heat") {
    predictors <- c("year", "tax", "ban", "heatpump", "energyclass", "exemption")
  } else if (experiment == "pv") {
    predictors <- c("mix", "imports", "pv", "tradeoffs", "distribution")
  } else {
    stop("Error: experiment must be either 'heat' or 'pv'.")
  }

  # create the formula dynamically
  formula <- as.formula(paste("Y ~", paste(predictors, collapse = " + ")))
  
  # run the conditional marginal means (cj) function
  mm_results <- cj(
    df,
    formula,
    id = ~ID,
    estimate = "mm",
    by = as.formula(paste("~", by))  # Group by variable passed as 'by'
  )
  
  # optionally save the file
  if (save_file) {
    # Save to CSV (you can choose any format or path)
    write.csv(mm_results, 
              file = paste0("data/",
                            experiment,
                            "_",
                            by,
                            "_MMs.csv"), 
              row.names = FALSE)
  }

  # initialize plot_output to NULL
  plot_output <- NULL

  # optionally plot results
  if (get_plot) {
    plot_output <- plot(mm_results, group = by, vline = 0.5) +
      labs(title = paste0("Choice outcome of ", by))
      # xlim(0.3, 0.7)
      # theme_nice()
    # return results with the plot data
    return(list(mm_results = mm_results, plot = plot_output))
  }

  else {
    return(list(mm_results = mm_results))
  }
}