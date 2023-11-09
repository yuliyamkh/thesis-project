library(tidyverse)
library(argparse)
library(ggplot2)
library(patchwork)
library(car)
library(effects)
library(performance)


parser <- ArgumentParser()
parser$add_argument("--path", help="Path to the CSV file")
args <- parser$parse_args()

# HELPER FUNCTIONS
# Read data from a given file path
read_csv_file <- function(path) {
  data <- read.csv(path)
  return(data)
}

# Check for linearity
check_linearity <- function(data, x_var, y_var) {
  ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point() +
    geom_smooth(method = 'lm') +
    geom_smooth(method = 'loess', colour='firebrick4') +
    labs(y = expression(italic('L')), x = expression(paste("log"[10], "(", italic("N"), ")"))) +
    theme_minimal()
}

# Check for normality
check_normality <- function(mdl) {
  residuals <- resid(mdl)
  empirical_density <- density(residuals)
  theoretical_density <- dnorm(empirical_density$x, mean = 0, sd = sd(residuals))
  
  ggplot() +
    geom_density(data = data.frame(residuals = residuals), aes(x = residuals, y = ..density..), fill = "lightblue", alpha = 0.6) +
    geom_line(data = data.frame(x = empirical_density$x, y = empirical_density$y), aes(x = x, y = y), color = "blue") +
    geom_line(data = data.frame(x = empirical_density$x, y = theoretical_density), aes(x = x, y = y), color = "red", linetype = "dashed") +
    labs(x = "Residuals", y = "Density") +
    theme_minimal()
}

# Check for homoscedasticity
check_homoscedasticity <- function(mdl) {
  residuals <- resid(mdl)
  fitted_values <- predict(mdl)
  
  ggplot(data.frame(fitted = fitted_values, residuals = residuals), aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = TRUE, color = "blue", linetype = "solid") +
    labs(x = "Fitted",
         y = "Residuals") +
    theme_minimal()
}

# LOAD DATA
# generated under neutral change, replicator selection, and interactor selection
nc_data <- read_csv_file("C:\\Users\\avror\\PycharmProjects\\ABM_AgentPy\\output\\neutral_change_output.csv")
rs_data <- read_csv_file("C:\\Users\\avror\\PycharmProjects\\ABM_AgentPy\\output\\replicator_selection_output.csv")
is_data <- read_csv_file("C:\\Users\\avror\\PycharmProjects\\ABM_AgentPy\\output\\interactor_selection_output.csv")

head(nc_data)
head(rs_data)
head(is_data)

# DATA TRANSFORMATION
# neutral change
nc_data <- mutate(nc_data,
                  LogPopulationSize = log10(population_size),
                  RewProbFactor = factor(rewiring_probability))

# replicator selection
rs_data <- mutate(rs_data,
                  LogPopulationSize = log10(population_size),
                  RewProbFactor = factor(rewiring_probability),
                  OrdinalPressure = factor(selection_pressure, ordered = TRUE))

contrasts(rs_data$OrdinalPressure) <- contr.treatment(10)

# interactor selection
is_data <- mutate(is_data,
                  LogPopulationSize = log10(population_size),
                  RewProbFactor = factor(rewiring_probability),
                  OrdinalPressure = factor(selection_pressure, ordered = TRUE),
                  Leaders = factor(n, ordered = TRUE))

contrasts(is_data$OrdinalPressure) <- contr.treatment(10)
contrasts(is_data$Leaders) <- contr.treatment(2)

# MODEL RUN
# neutral change
M_nc <- lm(final_A ~ LogPopulationSize + RewProbFactor, data = nc_data)
# replicator selection
M_rs <- lm(final_A ~ LogPopulationSize + RewProbFactor + OrdinalPressure, data = rs_data)
# interactor selection
M_is <- lm(final_A ~ LogPopulationSize + RewProbFactor + OrdinalPressure + Leaders, data = is_data)

# MODEL OUPUTS
summary(M_nc)
summary(M_rs)
summary(M_is)

# ASSUMPTIONS
# 1. Linearity
lin_nc <- check_linearity(nc_data, "LogPopulationSize", "final_A")
lin_rs <- check_linearity(rs_data, "LogPopulationSize", "final_A")
lin_is <- check_linearity(is_data, "LogPopulationSize", "final_A")

empty_plot <- plot_spacer()
combined_lin <- (lin_nc | lin_rs) / (lin_is | empty_plot)
print(combined_lin)

# 2. Normality
norm_nc <- check_normality(M_nc)
norm_rs <- check_normality(M_rs)
norm_is <- check_normality(M_is)

empty_plot <- plot_spacer()
combined_lin <- (norm_nc | norm_rs) / (norm_is | empty_plot)
print(combined_lin)

# 3. Homoscedasticity
hom_nc <- check_homoscedasticity(M_nc)
hom_rs <- check_homoscedasticity(M_rs)
hom_is <- check_homoscedasticity(M_is)

empty_plot <- plot_spacer()
combined_lin <- (hom_nc | hom_rs) / (hom_is | empty_plot)
print(combined_lin)

# 4. Multicollinearity
vif_nc <- vif(M_nc) # low correlation
vif_rs <- vif(M_rs) # low correlation
vif_is <- vif(M_is) # low correlation

# OUTLIERS
check_outliers(M_nc) # 2 outliers detected: cases 151, 302
check_outliers(M_rs) # No outliers detected
check_outliers(M_is) # No outliers detected

# EFFECT SIZES
plot(predictorEffects(M_nc))
plot(predictorEffects(M_rs))
plot(predictorEffects(M_is))

