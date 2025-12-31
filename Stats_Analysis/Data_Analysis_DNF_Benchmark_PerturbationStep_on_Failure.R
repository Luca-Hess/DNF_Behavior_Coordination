
rm(list=ls())

# Loading libraries
library(tidyverse)
library(emmeans)
library(car)
library(DescTools)
library(epitools)
library(pwr)

# Loading data
df_robust <- read_csv("bt_dnf_benchmark_results_robustness.csv")

###### 
# Data Preparation
######

# Reshaping data to long format 
df_robust_long <- df_robust %>%
  pivot_longer(
    cols = c(`BT Time (s)`, `BT Steps`, `BT Success`,
             `SM Time (s)`, `SM Steps`, `SM Success`,
             `DNF Time (s)`, `DNF Steps`, `DNF Success`),
    names_to = c("Paradigm", ".value"),
    names_pattern = "(BT|SM|DNF)\\s*(.*)"
  )

# Renaming columns
colnames(df_robust_long)[5] <- "Time"
colnames(df_robust_long)[3] <- "Perturbation_Step"

# Changing appropriate columns to factors
df_robust_long$Paradigm <- as.factor(df_robust_long$Paradigm)
df_robust_long$Perturbation <- as.factor(df_robust_long$Perturbation)

##########################################
# Effect of Perturbation Step on Success #
##########################################
df_robust_long$Success <- as.factor(df_robust_long$Success)

# Checking data requirements:
OD_subset <- subset(df_robust_long, Perturbation == "Object Displacement")
SN_subset <- subset(df_robust_long, Perturbation == "Sensor Noise")

table(OD_subset$Paradigm,
      OD_subset$Perturbation_Step,
      OD_subset$Success)

table(SN_subset$Paradigm,
      SN_subset$Perturbation_Step,
      SN_subset$Success)

# Plenty of 0-cells, meaning this data is not suitable for GLM analysis.
# Proceeding with Cochran-Armitage trend test instead.

# Assumptions of Cochran-Armitage trend test:
# 1. The response variable is binary (Success: 0 or 1) - met
# 2. The predictor variable is ordinal (Perturbation Step) - met
# 3. The observations are independent - met
# 4. Sufficient sample size - Not met! Sparse data because we only have 
# one observation per perturbation step per combination of Paradigm and Perturbation.

# Due to data sparsity, we will pool the data into bins (early, middle, late)
# The bins are based on the simulation, where perturbation was set to be able 
# to occur between steps 200 and 1200.

breaks <- c(200, 533, 866, 1200)
labels <- c("Early", "Middle", "Late")

df_robust_long_binned <- df_robust_long %>%
  mutate(step_bin = cut(as.numeric(Perturbation_Step),
                        breaks = breaks,
                        labels = labels,
                        include.lowest = TRUE,
                        right = FALSE))

OD_subset <- subset(df_robust_long_binned, Perturbation == "Object Displacement")
SN_subset <- subset(df_robust_long_binned, Perturbation == "Sensor Noise")



# Creating separate table for various paradigms and perturbation steps
tab_DNF_OD <- table(OD_subset$Success[OD_subset$Paradigm == "DNF"],
                    OD_subset$step_bin[OD_subset$Paradigm == "DNF"])

tab_BT_OD <- table(OD_subset$Success[OD_subset$Paradigm == "BT"],
                   OD_subset$step_bin[OD_subset$Paradigm == "BT"])

tab_SM_OD <- table(OD_subset$Success[OD_subset$Paradigm == "SM"],
                   OD_subset$step_bin[OD_subset$Paradigm == "SM"])


tab_DNF_SN <- table(SN_subset$Success[SN_subset$Paradigm == "DNF"],
                    SN_subset$step_bin[SN_subset$Paradigm == "DNF"])

tab_BT_SN <- table(SN_subset$Success[SN_subset$Paradigm == "BT"],
                   SN_subset$step_bin[SN_subset$Paradigm == "BT"])

tab_SM_SN <- table(SN_subset$Success[SN_subset$Paradigm == "SM"],
                   SN_subset$step_bin[SN_subset$Paradigm == "SM"])

# For whole perturbation type (MULTIPLE TESTING AT THIS POINT!)
tab_OD <- table(OD_subset$Success,
                OD_subset$step_bin)

tab_SN <- table(SN_subset$Success,
                SN_subset$step_bin)

# Performing Tests

# Object Displacement - separate paradigms
dnf_od_test <- CochranArmitageTest(tab_DNF_OD, alternative = "two.sided")
bt_od_test <- CochranArmitageTest(tab_BT_OD, alternative = "two.sided")
sm_od_test <- CochranArmitageTest(tab_SM_OD, alternative = "two.sided")

# Sensor Noise - separate paradigms
dnf_sn_test <- CochranArmitageTest(tab_DNF_SN, alternative = "two.sided")
bt_sn_test <- CochranArmitageTest(tab_BT_SN, alternative = "two.sided")
sm_sn_test <- CochranArmitageTest(tab_SM_SN, alternative = "two.sided")

# Object Displacement - all paradigms
od_test <- CochranArmitageTest(tab_OD, alternative = "two.sided")

# Sensor Noise - all paradigms
sn_test <- CochranArmitageTest(tab_SN, alternative = "two.sided")

# Raw Results - Negative trend = more failures for later perturbations
dnf_od_test # significant NEGATIVE trend (p < 0.01)
bt_od_test  # significant NEGATIVE trend (p < 0.001)
sm_od_test  # significant POSITIVE trend (p = 0.02)

dnf_sn_test # no significant trend (p = 0.07)
bt_sn_test  # no significant trend (p = 0.66)
sm_sn_test  # no significant trend (p = 0.44)

od_test     # significant NEGATIVE trend (p = 0.02)

sn_test     # no significant trend (p = 0.08)

# Collecting P values
pvals_primary <- c(
  DNF_OD = dnf_od_test$p.value,
  BT_OD =  bt_od_test$p.value,
  SM_OD =  sm_od_test$p.value,
  DNF_SN = dnf_sn_test$p.value,
  BT_SN =  bt_sn_test$p.value,
  SM_SN =  sm_sn_test$p.value)

p_adjusted_primary <- p.adjust(pvals_primary, method = "holm")

results_primary <- data.frame(
  Test = names(pvals_primary),
  p_raw = pvals_primary,
  p_adjusted = p_adjusted_primary,
  Significant = p_adjusted_primary < 0.05
)


# PURELY EXPLORATORY - not reported in thesis 
pvals_secondary <- c(
  OD = od_test$p.value,
  SN = sn_test$p.value)

results_secondary <- data.frame(
  Test = names(pvals_secondary),
  p_raw = pvals_secondary,
  Significant = pvals_secondary < 0.05)



# Validating approach

# Function to check cell counts
check_table <- function(tab, name) {
  cat("\n", name, ":\n")
  print(tab)
  cat("Row totals:", rowSums(tab), "\n")
  cat("Column totals:", colSums(tab), "\n")
  cat("Expected counts under independence:\n")
  print(chisq.test(tab)$expected)
  cat("Min expected count:", min(chisq.test(tab)$expected), "\n")
  
  # Rule of thumb: all expected counts should be >= 5
  if(min(chisq.test(tab)$expected) < 5) {
    cat("WARNING: Some expected counts < 5\n")
  } else {
    cat("All expected counts >= 5\n")
  }
}

check_table(tab_DNF_OD, "DNF Object Displacement")
check_table(tab_BT_OD, "BT Object Displacement")
check_table(tab_SM_OD, "SM Object Displacement")
check_table(tab_DNF_SN, "DNF Sensor Noise")
check_table(tab_BT_SN, "BT Sensor Noise")
check_table(tab_SM_SN, "SM Sensor Noise")

check_table(tab_OD, "All Paradigms Object Displacement")
check_table(tab_SN, "All Paradigms Sensor Noise")
# All tables had cells with low (n<5) data counts, indicating possibly lacking 
# statistical power. Moreover, the 20 runs in the OD perturbation regime have,
# by chance, generated more runs in the 200-533 step bin than in the other bins
# combined, skewing the results. 

# Checking effect sizes
# calculating r from test statistic: r  = Z / sqrt(N)

calculate_effectsize <- function(test_result, n) {
  z <- test_result$statistic
  r <- z / sqrt(n)
  return(r)
}

r_values <- c(
  DNF_OD = calculate_effectsize(dnf_od_test, sum(tab_DNF_OD)),
  BT_OD = calculate_effectsize(bt_od_test, sum(tab_BT_OD)),
  SM_OD = calculate_effectsize(sm_od_test, sum(tab_SM_OD)),
  DNF_SN = calculate_effectsize(dnf_sn_test, sum(tab_DNF_SN)),
  BT_SN = calculate_effectsize(bt_sn_test, sum(tab_BT_SN)),
  SM_SN = calculate_effectsize(sm_sn_test, sum(tab_SM_SN))
)

results_primary$Effect_Size_r <- r_values

# r < 0.1: negligible
# 0.1 <= r < 0.3: small
# 0.3 <= r < 0.5: medium
# r >= 0.5: large

# Power analysis
power_analysis <- function(n, r, alpha = 0.05/6) {
  power_result <- pwr.r.test(n = n, r = r, sig.level = alpha)
  return(power_result$power)
}

n_vals = c(
  DNF_OD = sum(tab_DNF_OD),
  BT_OD = sum(tab_BT_OD),
  SM_OD = sum(tab_SM_OD),
  DNF_SN = sum(tab_DNF_SN),
  BT_SN = sum(tab_BT_SN),
  SM_SN = sum(tab_SM_SN)
)

results_primary$Power <- mapply(power_analysis,
                                n_vals,
                                abs(results_primary$Effect_Size_r))

results_primary$Adequate_Power <- results_primary$Power >= 0.8

# Descriptive statistics
extract_success_rates <- function(tab) {
  props <- prop.table(tab, 2)[2, ]  # Success rate per bin
  list(
    early = props["Early"],
    middle = props["Middle"],
    late = props["Late"]
  )
}

success_rates <- list(
  extract_success_rates(tab_DNF_OD),
  extract_success_rates(tab_BT_OD),
  extract_success_rates(tab_SM_OD),
  extract_success_rates(tab_DNF_SN),
  extract_success_rates(tab_BT_SN),
  extract_success_rates(tab_SM_SN)
)

results_primary$Success_Early <- sapply(success_rates, function(x) sprintf("%.0f%%", x$early * 100))
results_primary$Succes_Middle <- sapply(success_rates, function(x) sprintf("%.0f%%", x$middle * 100))
results_primary$Success_Late <- sapply(success_rates, function(x) sprintf("%.0f%%", x$late * 100))

# Reorder columns
results_primary <- results_primary[, c("Test", "Success_Early", "Succes_Middle",
                                       "Success_Late", "Effect_Size_r", "p_raw", 
                                       "p_adjusted", "Significant", "Power", 
                                       "Adequate_Power")]



# Results
results_primary

results_secondary




# Data did not meet the assumptions for GLM analysis due to sparsity.
# As such, I have performed Cochran-Armitage trend tests instead.
# The results indicate significant trends in some cases, but even with binning into 
# three categories (early, middle, late), the data remains sparse, 
# leading to low counts in several cells and reducing statistical power.
# Moreover, since multiple tests were performed, the p-values needed to be 
# adjusted, for which I used the Holm method.
# Not shown here is an exploration where data was instead binned into two categories,
# late and early. This did lead to higher counts per cell, and somewhat adressed
# the imbalance that is present (particularly in the OD setting), where more
# runs had early perturbation steps than late ones. However, this approach
# did not at all change the outcome, with the same tests being significant
# and having adequate power, which is why this analysis was not pursued further.
# The final results show significant trends for DNF (p = 0.03, r = 0.62, large effect)
# and BT (p < 0.001, r = 0.88, large effect) paradigms under the object displacement 
# perturbation. The SM data did not show such a trend (p = 0.06, r = 0.54, large effect).
# The two significant results indicated an increase in simulation failures
# with later perturbation steps. 
# In the sensor noise perturbation setting, there was no evidence for significant
# trends in success rate (DNF: p = 0.21, r = 0.41, medium effect, BT: p = 0.88, 
# r = 0.1, small effect, SM: p = 0.17, r = 0.17, small effect).
# However, what trends we could observe visually as well as based on the sign of 
# the effect size, indicate that sensor noise might trend in he opposite direction, 
# with later perturbations leading to fewer failures. The robot has a 
# second simulated sensor in its gripper, which allows it to confirm if it has 
# an object grabbed. Data from this sensor overwrites the visually acquired 
# object location, which is the one subjected to perturbation. As such, these
# findings make sense, though I reiterate that none were significant.
# Power analysis showed that only one of these six tests actually had adequate
# power, this being the BT object displacement trend. 
# In the end, we can report, that there appear to be some trends indicating that a 
# later object displacement leads to more simulation failures, a finding that
# is consistent with expectations but limited by low statistical power. 
# Functionally, this is the difference between simply detecting, 
# orienting or moving to a different targt location (early perturbation),
# or basically losing a grasped object and needing to redect, relocate and regrab it.
# In short, the number of steps for recovery is larger with a late perturbation.
# Sensor noise perturbations on the other hand do not appear to be subject to such trends,
# though here the low power requires bearing in mind that the absence of evidence 
# is not necessarily the evidence of absence of such an effect.
# Additionally, I performed purely exploratory analysis of pooled data for the
# two perturbation schemes. As the holm corrected analyses of the paradigms had
# indicated, the object displacement perturbation seems subject to a significant
# trend in success rate, where later perturbations lead to a reduction.
# And similarly, the sensor noise perturbation data do not show significant trends.



###############
# Visualization
###############

plot_data <- df_robust_long_binned %>%
  group_by(Paradigm, Perturbation, step_bin) %>%
  summarise(success_rate = mean(as.logical(Success)),
            n = n(),
            .groups = "drop")

ggplot(plot_data, aes(x = step_bin, 
                      y = success_rate, 
                      color = Paradigm, 
                      group = Paradigm)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~Perturbation) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(title = "Effect of Perturbation Timing on Success Rate",
       y = "Success Rate", x = "Perturbation Timing") +
  theme_classic()

#############
# Final Table
#############

results_primary

