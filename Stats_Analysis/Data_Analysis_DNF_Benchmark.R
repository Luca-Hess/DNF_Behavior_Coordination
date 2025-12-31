
rm(list=ls())

# Loading libraries
library(tidyverse)
library(effectsize)
library(emmeans)
library(ARTool)
library(car)
library(broom)

citation('tidyverse')
citation('effectsize')
citation('emmeans')
print(citation('ARTool'), bibtex=TRUE)
citation('car')
citation('broom')
citation()
RStudio.Version()

# Loading data
df_speed <- read_csv("bt_dnf_benchmark_results_speed.csv")

df_robust <- read_csv("bt_dnf_benchmark_results_robustness.csv")

################################################################################
# Speed analysis
###################

###### 
# Data Preparation
######

# Taking out data that consists only of the same value
bt_steps <- unique(df_speed$`BT Steps`)
sm_steps <- unique(df_speed$`SM Steps`)
dnf_steps <- unique(df_speed$`DNF Steps`)

# Drop those columns
df_speed <- df_speed %>%
  select(-`BT Steps`, -`SM Steps`, -`DNF Steps`,
         -`BT Success`, -`SM Success`, -`DNF Success`, 
         - `DNF Theoretical Time (s)`, - `Run`)

# Reshaping data to long format 
df_speed_long <- df_speed %>%
  pivot_longer(cols = everything(),
               names_to = c("Paradigm"),
               values_to = "Value")

# Renaming paradigms
df_speed_long$Paradigm <- recode_factor(df_speed_long$Paradigm,
                                        `BT Time (s)` = "BT",
                                        `SM Time (s)` = "SM",
                                        `DNF Time (s)` = "DNF")

######
# Data Analysis & Plotting
######

# Summary statistics
df_speed_summary <- df_speed_long %>%
  group_by(Paradigm) %>%
  summarise(Mean = mean(Value),
            SD = sd(Value),
            Median = median(Value),
            IQR = IQR(Value))
print(df_speed_summary)

# Boxplot - making sure to start y axis at 0
ggplot(df_speed_long, aes(x = Paradigm, y = Value)) +
  geom_boxplot() +
  labs(title = "Execution Time Comparison",
       x = "Paradigm",
       y = "Execution Time (s)") +
  theme_classic() +
  ylim(0, NA) # Set y-axis to start at 0

ggplot(df_speed_long, aes(x = Paradigm, y = Value)) +
  geom_boxplot() +
  labs(title = "Execution Time Comparison",
       x = "Paradigm",
       y = "Execution Time (s)") +
  theme_classic() 


ggplot(df_speed_summary, aes(x = Paradigm, y = Mean)) +
  geom_bar(stat = "identity", 
           position = position_dodge(0.7), 
           fill='white', 
           color='black',
           width=0.6) +
  geom_errorbar(aes(ymin=Mean - SD, ymax=Mean + SD),
                width=0.2,
                position=position_dodge(0.7)) +
  labs(title = "Execution Time Comparison",
       x = "Paradigm",
       y = "Execution Time (s)") +
  theme_classic() +
  ylim(0, NA) # Set y-axis to start at 0
######
# Statistical Testing
######

# Checking normality assumption with Shapiro-Wilk test and plots
shapiro_results <- df_speed_long %>%
  group_by(Paradigm) %>%
  summarise(Shapiro_p = shapiro.test(Value)$p.value)
print(shapiro_results)
? shapiro.test
# All shapiro p > 0.05, indicating normality


# QQ Plots
par(mfrow=c(1,3))
for(paradigm in unique(df_speed_long$Paradigm)) {
  qqnorm(df_speed_long$Value[df_speed_long$Paradigm == paradigm],
         main = paste("QQ Plot -", paradigm))
  qqline(df_speed_long$Value[df_speed_long$Paradigm == paradigm])
}
par(mfrow=c(1,1))

# Not as clear cut, but still no gross problems. It seems fair to assume data
# are normally distributed.


# Checking Homoscedasticity
levene_result <- car::leveneTest(Value ~ Paradigm, data = df_speed_long)
print(levene_result)
# Levene's test p > 0.05, indicating homoscedasticity

# Plotting residuals
# Residuals vs Fitted plot
plot(aov(Value ~ Paradigm, data = df_speed_long), which = 1)
# Looks acceptable, the variance of the residuals appears roughly constant across 
# fitted values. Notable is upwards trend in residuals at higher fitted values.
# Still, combined with the Levene's test, I will assume the assumption of 
# homoscedasticity is met.

# Residuals QQ plot
plot(aov(Value ~ Paradigm, data = df_speed_long), which = 2)
# Shapiro-Wilk test on residuals
shapiro.test(residuals(aov(Value ~ Paradigm, data = df_speed_long)))

# Does not raise any cause for concern, the residuals appear normally distributed.

# Leverage plot
plot(aov(Value ~ Paradigm, data = df_speed_long), which = 5)
# No points of major concern here either.



# ANOVA Test
# Setting contrasts
options(contrasts = c("contr.sum", "contr.poly"))

anova_result <- aov(Value ~ Paradigm, data = df_speed_long)
summary(anova_result)

# The main effect Paradigm is significant (p < 0.01, F(2,57) = 569.7),
# indicating that the different paradigms have significantly different 
# execution times.


# Post-hoc Tukey HSD test
tukey_result <- TukeyHSD(anova_result)
print(tukey_result)

# Significant impact of paradigm on execution time, with significant pairwise
# comparisons. BT is fastest, followed by SM and DNF.

# Effect size analysis
# Eta squared for ANOVA
eta_squared_result <- eta_squared(anova_result)
print(eta_squared_result)

# Most of the variance (95%) explained by paradigm.
# Due to the setup, it stands to reason that the remaining variance
# is due to changes in system load of the machine running the benchmarks.
# We can report there is a large effect (eta^2 = 0.95) of paradigm choice 
# on execution time.

# Omega squared for ANOVA
omega_squared_result <- omega_squared(anova_result)
print(omega_squared_result)
# Omega squared further supports the finding of a large effect size (omega^2 = 0.95).

# Cohen's d for pairwise comparisons
d_SM_DNF <- cohens_d(Value ~ Paradigm, 
                     data = subset(df_speed_long, Paradigm %in% c("SM","DNF")))

d_SM_BT <- cohens_d(Value ~ Paradigm, 
                    data = subset(df_speed_long, Paradigm %in% c("SM","BT")))

d_DNF_BT <- cohens_d(Value ~ Paradigm, 
                     data = subset(df_speed_long, Paradigm %in% c("DNF","BT")))

print(d_SM_DNF)
print(d_SM_BT)
print(d_DNF_BT)
# We find effect sizes of d = 1.11 (SM vs DNF), d = -9.2 (SM vs BT),
# and d = -9.98 (DNF vs BT), indicating large effects (d > 0.8) in all comparisons.
# We can however note, that the effect size of the SM vs DNF comparison
# was smallest, which is reflected in plots of the data.
# Moreover, we note that the values for Cohen's d are very large,
# likely due to the small standard deviations of the treatment groups 
# (SDs: BT = 0.061, DNF = 0.084, SM = 0.078).
# AS such, we supplement our reporting of Cohen's d with Cliff's delta

# Cliff's delta for pairwise comparisons
delta_SM_DNF <- cliffs_delta(Value ~ Paradigm, 
                             data = subset(df_speed_long, Paradigm %in% c("SM","DNF")))
delta_SM_BT <- cliffs_delta(Value ~ Paradigm, 
                            data = subset(df_speed_long, Paradigm %in% c("SM","BT")))
delta_DNF_BT <- cliffs_delta(Value ~ Paradigm,
                             data = subset(df_speed_long, Paradigm %in% c("DNF","BT")))
print(delta_SM_DNF)
print(delta_SM_BT)
print(delta_DNF_BT)
# Cliff's delta is more robust to small SDs than Cohen's d, hence it is an
# appropriate supplementary effect size measure in this case.
# The results further support the finding of large effect sizes (delta > 0.47) for 
# each paradigm comparison (delta = 0.54 for SM vs DNF, delta = -1 for SM vs BT,
# and delta = -1 for DNF vs BT).

# In conclusion we can report that there is a significant effect of paradigm choice
# on benchmark completion time (p<0.01, F(2,57)=569.7, eta^2=0.95, large effect).
# Post-hoc tests indicate significant pairwise differences between all paradigms,
# with Behavior Trees (mean 7.98s, 0.06 SD)completing the fastest, followed by the Hierarchical
# State Machine (mean 8.62s, 0.08 SD) implementation, and finally our Dynamic Neural Fields model (mean 8.71s, 0.08s)
# (p<0.001, delta=1 DNF-BT, p<0.001, delta=1 SM-BT, p=0.001, delta=0.54 SM-DNF).
# All comparisons had large effect sizes (see above).
# The total number of benchmark trials performed was 60 (20 per paradigm).
# Moreover we can note that all BT runs completed in 1329 steps, all SM runs in 
# 1334 steps, and all DNF runs in 1916 steps, with all three paradigms 
# successfully completing all 20 benchmark runs.

#############
# Final Table
#############

# --- Tukey HSD results ---
tukey_summary <- as.data.frame(tukey_result$Paradigm)
tukey_table <- rownames_to_column(tukey_summary, "Comparison") %>%
  transmute(
    Comparison,
    p_value = signif(`p adj`, 3)
  )

# --- Pairwise effect sizes ---
effect_table <- tibble(
  Comparison = c("SM-BT", "DNF-BT", "DNF-SM" ),
  Cohens_d = round(c(d_SM_DNF$Cohens_d, d_SM_BT$Cohens_d, d_DNF_BT$Cohens_d), 2),
  Cliffs_delta = round(c(delta_SM_BT$r_rank_biserial, 
                         delta_DNF_BT$r_rank_biserial,
                         delta_SM_DNF$r_rank_biserial), 2)
)

# --- Merge Tukey + Effect sizes ---
pairwise_results <- left_join(tukey_table, effect_table, by = "Comparison")

# --- ANOVA summary row ---
anova_summary <- summary(anova_result)[[1]]
anova_row <- tibble(
  Comparison = "ANOVA (Main effect)",
  F_value = round(anova_summary[1, "F value"], 2),
  df1 = anova_summary[1, "Df"],
  df2 = anova_summary[2, "Df"],
  p_value = signif(anova_summary[1, "Pr(>F)"], 3),
  eta_sq = round(eta_squared_result$Eta2[1], 2),
  omega_sq = round(omega_squared_result$Omega2[1], 2),
  Cohens_d = NA,
  Cliffs_delta = NA
)

# --- Final combined table ---
results_table_speed <- bind_rows(anova_row, pairwise_results)

print(results_table_speed)



################################################################################
################################################################################
################################################################################
# Robustness analysis
###################

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


# Summary statistics
df_robust_summary <- df_robust_long %>%
  group_by(Paradigm, Perturbation) %>%
  summarise(Mean_t = mean(Time),
            SD_t = sd(Time),
            Median_t = median(Time),
            IQR_t = IQR(Time),
            Success_Rate = mean(Success),
            Successes = sum(Success),
            Avg_Steps = mean(Steps))
print(df_robust_summary)


#################
# Plotting Data
#################

ggplot(df_robust_long, aes(x = Paradigm, y = Time, fill=Perturbation)) +
  geom_boxplot() +
  labs(title = "Execution Time Comparison",
       x = "Paradigm",
       y = "Execution Time (s)") +
  theme_classic() +
  ylim(0, NA) # Set y-axis to start at 0

ggplot(df_robust_long, aes(x = Paradigm, y = Steps, fill=Perturbation)) +
  geom_boxplot() +
  labs(title = "Number of Steps Comparison",
       x = "Paradigm",
       y = "Number of Steps to Completion") +
  theme_classic() +
  ylim(0, NA) # Set y-axis to start at 0

# Supplementary plot after conflicting analysis on impact of perturbation on
# number of steps
ggplot(df_robust_long, aes(x = Perturbation, y = Steps)) +
  geom_boxplot() +
  labs(title = "Number of Steps Comparison by Perturbation",
       x = "Perturbation",
       y = "Number of Steps to Completion") +
  theme_classic() +
  ylim(0, NA) # Set y-axis to start at 0


ggplot(df_robust_summary, aes(x = Paradigm, y = Successes, fill = Perturbation)) +
  geom_bar(stat = "identity",
           position = position_dodge(0.7),
           color = 'black',
           width = 0.6) +
  geom_hline(aes(yintercept = mean(Successes), linetype = "Grand Mean"),
             color = "black") +
  scale_linetype_manual(name = "", values = c("Grand Mean" = "dashed")) +
  labs(title = "Success Comparison",
       x = "Paradigm",
       y = "Number of Successful Completions",
       fill = "Perturbation") +
  theme_classic() +
  ylim(0, NA) # Set y-axis to start at 0


######
# Statistical Testing
######

########################################
# Completion Speed under Perturbations #
########################################

# Checking normality assumption with Shapiro-Wilk test and plots
shapiro_results_time <- df_robust_long %>%
  group_by(Paradigm, Perturbation) %>%
  summarise(Shapiro_p = shapiro.test(Time)$p.value)
print(shapiro_results_time)

# Most shapiro p < 0.05, indicating non-normality


# QQ Plots
par(mfrow=c(2,3))  # 2 rows, 3 columns

for (paradigm in unique(df_robust_long$Paradigm)) {
  for (perturb in unique(df_robust_long$Perturbation)) {
    
    subset_data <- df_robust_long$Time[
      df_robust_long$Paradigm == paradigm & 
        df_robust_long$Perturbation == perturb
    ]
    
    qqnorm(subset_data,
           main = paste("QQ Plot -", paradigm, "-", perturb))
    qqline(subset_data)
  }
}

par(mfrow=c(1,1))  # reset layout

# These QQ plots also indicate strong deviations from normality.


# Checking Homoscedasticity
levene_result_time <- car::leveneTest(Time ~ Paradigm * Perturbation, data = df_robust_long)
print(levene_result_time)
# Levene's test p > 0.05, indicating homoscedasticity

# Plotting residuals
# Residuals vs Fitted plot
plot(aov(Time ~ Paradigm * Perturbation, data = df_robust_long), which = 1)
# This plot indicates problems with homoscedasticity, as the variance of the
# residuals generally seems to increase with fitted values.
# Though this is only applicable for a few outlier values, I do not feel comfortable
# considering the assumption of homoscedasticity to be met. 

# Residuals QQ plot
plot(aov(Time ~ Paradigm * Perturbation, data = df_robust_long), which = 2)
shapiro.test(residuals(aov(Time ~ Paradigm * Perturbation, data = df_robust_long)))
# Both the QQ Plot and the Shapiro-Wilk test indicate deviations from normality.

# Based on the above analysis, I conclude that we must use robust methods for 
# this analysis.

# Leverage plot
plot(aov(Time ~ Paradigm * Perturbation, data = df_robust_long), which = 5)
# possibly some 

# Robust Aligned Rank Transform (ART) ANOVA
# Setting contrasts
options(contrasts = c("contr.sum", "contr.poly"))

model_art <- art(Time ~ Paradigm * Perturbation, data = df_robust_long)
anova(model_art)

# The main effect Paradigm is significant (p < 0.01, F(2,114) = 5.43),
# indicating that the different paradigms have different 
# execution times. However, which perturbation is applied does not significantly
# affect execution time (p = 0.34, F(1, 114) = 0.92).
# Finally, there is no significant interaction effect between Paradigm and 
# Perturbation, indicating that the paradigms do not react differently to the 
# perturbations in terms of their execution time (p = 0.9, F(2,114) = 0.1).


# Post-hoc test
post_hoc_time <- pairwise.wilcox.test(df_robust_long$Time, df_robust_long$Paradigm,
                                      p.adjust.method = "holm")
print(post_hoc_time)
# Non-parametric post-hoc analysis (pairwise Wilcox tests using Holm correction)
# reveals that Behavior Trees are significantly faster than 
# Hierarchical State Machines (p < 0.01). However, there are no significant
# differences between Behavior Trees and Dynamic Neural Fields (p = 0.68) or
# between Hierarchical State Machines and Dynamic Neural Fields (p = 0.12).

# Effect size analysis
# partial Eta squared for ANOVA
eta_squared_result <- eta_squared(anova(model_art))
print(eta_squared_result)

# Most of the variance (9%) in the ART transformed data is explained by paradigm.
# We can report there is a medium effect (eta^2 = 0.09) of paradigm choice 
# on the transformed execution time data (0.06 < eta^2 < 0.14). The other effects
# are minuscule (eta^2 < 0.01)


# Cliff's delta for pairwise comparisons
delta_SM_DNF <- cliffs_delta(Time ~ Paradigm, 
                             data = subset(df_robust_long, Paradigm %in% c("SM","DNF")))
delta_SM_BT <- cliffs_delta(Time ~ Paradigm, 
                            data = subset(df_robust_long, Paradigm %in% c("SM","BT")))
delta_DNF_BT <- cliffs_delta(Time ~ Paradigm,
                             data = subset(df_robust_long, Paradigm %in% c("DNF","BT")))
print(delta_SM_DNF)
print(delta_SM_BT)
print(delta_DNF_BT)
# Cohen's d is inappropriate in a non-parametric setting, which is why we omit this
# analysis in this case.
# Cliff's delta is more robust to small SDs and non-parametric data 
# than Cohen's d, hence it is an appropriate effect size measure in this case.
# We find a small effect size (0.15 < delta < 0.33) for the comparisons 
# between SM and DNF (delta = 0.25), medium-near-large effect size (0.33 < delta < 0.47)
# for the comparison between SM and BT (delta = 0.46), and finally a negligible 
# effect size (delta < 0.147) for the comparison between DNF and BT (delta = 0.05).


# In conclusion we can report that there is a significant effect of paradigm choice
# on benchmark completion time (p<0.01, F(2,114)=5.43, eta^2=0.09 in ART
# transformed data, medium effect). The data don't show perturbation type and the interaction 
# between perturbation and the paradigm to have a significant effect on completion
# time.
# Post-hoc tests indicate significant pairwise differences in completion time
# between Behavior Trees and Hierarchical State Machines (p=0.001, delta=0.46, 
# medium effect), but not between Behavior Trees and Dynamic Neural Fields 
# (p=0.68, delta=0.05, negligible effect) nor between Hierarchical State Machines
# and Dynamic Neural Fields (p=0.12, delta=0.25, small effect).
# While perturbations do not have a significant effect on completion time,
# we can infer some effect by the fact that the separation between the three
# paradigms is not as clear as it was when comparing speed without any perturbations.
# Unsurprisingly, adding perturbations has increased the variance in the system.
# The total number of benchmark trials performed was 120 (20 per paradigm and perturbation).

#############
# Final Table
#############

# --- ART ANOVA summary row ---
anova_art <- anova(model_art)

anova_row <- tibble(
  Comparison = "ART ANOVA (Main effect: Paradigm)",
  F_value = round(anova_art$`F value`[1], 2),
  df1 = anova_art$Df[1],
  df2 = anova_art$Df.res[1],
  p_value = signif(anova_art$`Pr(>F)`[1], 3),
  eta_sq = round(eta_squared_result$Eta2[1], 2),   # from effectsize::eta_squared(art_model)
  Cliffs_delta = NA
)

# --- Pairwise post-hoc results (Wilcoxon) ---
wilcox_mat <- post_hoc_time$p.value
wilcox_table <- as.data.frame(wilcox_mat) %>%
  rownames_to_column("Group1") %>%
  pivot_longer(-Group1, names_to = "Group2", values_to = "p_value") %>%
  filter(!is.na(p_value)) %>%
  mutate(Comparison = paste(Group1, Group2, sep = "-"),
         p_value = signif(p_value, 3))


# --- Pairwise effect sizes ---
effect_table <- tibble(
  Comparison = c("DNF-BT", "SM-BT","SM-DNF"),
  Cliffs_delta = round(c(delta_DNF_BT$r_rank_biserial,
                         delta_SM_BT$r_rank_biserial,
                         delta_SM_DNF$r_rank_biserial), 2)
)

# --- Merge post-hoc + effect sizes ---
pairwise_results <- left_join(wilcox_table, effect_table, by = "Comparison")

# --- Final combined table ---
results_table <- bind_rows(anova_row, pairwise_results)
results_table_time <- results_table[-c(8,9)]

print(results_table_time)


#######################################
# Number of Steps under Perturbations #
#######################################

# Checking normality assumption with Shapiro-Wilk test and plots
shapiro_results_time <- df_robust_long %>%
  group_by(Paradigm, Perturbation) %>%
  summarise(Shapiro_p = shapiro.test(Steps)$p.value)
print(shapiro_results_time)
# Most shapiro p < 0.05, indicating non-normality


# QQ Plots
par(mfrow=c(2,3))  # 2 rows, 3 columns

for (paradigm in unique(df_robust_long$Paradigm)) {
  for (perturb in unique(df_robust_long$Perturbation)) {
    
    subset_data <- df_robust_long$Steps[
      df_robust_long$Paradigm == paradigm & 
        df_robust_long$Perturbation == perturb
    ]
    
    qqnorm(subset_data,
           main = paste("QQ Plot -", paradigm, "-", perturb))
    qqline(subset_data)
  }
}

par(mfrow=c(1,1))  # reset layout

# These QQ plots also indicate strong deviations from normality in some cases,
# others look mostly fine excepting some outlier values.


# Checking Homoscedasticity
levene_result_time <- car::leveneTest(Steps ~ Paradigm * Perturbation, data = df_robust_long)
print(levene_result_time)
# Levene's test p > 0.05, indicating homoscedasticity

# Plotting residuals
# Residuals vs Fitted plot
plot(aov(Steps ~ Paradigm * Perturbation, data = df_robust_long), which = 1)
# This plot indicates some problems with homoscedasticity, as the variance of the
# residuals generally seems to increase then decrease with fitted values.
# It is not straight-forward, but I don't feel comfortable considering the assumption
# of homoscedasticity to be met.

# Residuals QQ plot
plot(aov(Steps ~ Paradigm * Perturbation, data = df_robust_long), which = 2)
shapiro.test(residuals(aov(Steps ~ Paradigm * Perturbation, data = df_robust_long)))
# Both the QQ Plot and the Shapiro-Wilk test indicate deviations from normality.

# Based on the above analysis, I conclude that we must use robust methods for 
# this analysis.

# Leverage
plot(aov(Steps ~ Paradigm * Perturbation, data = df_robust_long), which = 5)

# Robust Aligned Rank Transform (ART) ANOVA
# Setting contrasts
options(contrasts = c("contr.sum", "contr.poly"))

model_art <- art(Steps ~ Paradigm * Perturbation, data = df_robust_long)
anova(model_art)

# The main effect Paradigm is significant (p << 0.01, F(2,114) = 13.15),
# indicating that the different paradigms differ in the number of steps performed
# until completion. Which perturbation is applied significantly affects the number
# steps performed until completion (p = 0.01, F(1, 114) = 6.58).
# However, there is no significant interaction effect between Paradigm and 
# Perturbation, indicating that the paradigms do not react differently to the 
# perturbations with regards to the number of steps needed to complete
# a simulation (p = 0.25, F(2,114) = 1.39).


# Post-hoc test
post_hoc_steps <- pairwise.wilcox.test(df_robust_long$Steps, df_robust_long$Paradigm,, data = df_robust_long,
                                       p.adjust.method = "holm")

print(post_hoc_steps)
# Non-parametric post-hoc analysis (pairwise Wilcox tests using Holm correction)
# reveals that both Behavior Trees (p < 0.01) and Hierarchical State Machines
# (p < 0.01) take significantly fewer steps than our Dynamic Neural Field system. 
# The number of steps between BTs and SMs is however not significantly different (p = 0.45).

steps_perturbation <- wilcox.test(Steps ~ Perturbation, data = df_robust_long,  
                                  p.adjust.method = "holm")
print(steps_perturbation)
# Non-parametric test (Wilcoxon) reveals that the number of steps performed
# between the object displacement and sensor glitch perturbations
# does not differ significantly (W=2037.5, p=0.21).

# Effect size analysis
# partial Eta squared for ANOVA
eta_squared_result <- eta_squared(anova(model_art))
print(eta_squared_result)

# Most of the variance (19%) in the ART transformed data is explained by paradigm.
# We can report there is a large effect (eta^2 = 0.19) of paradigm choice 
# on the transformed "number of steps" data (0.14 < eta^2). 
# The effect sizes of perturbation alone (eta^2 = 0.05) is close to medium (eta^2 = 0.06 is medium),
# that of the interaction (eta^2 = 0.02) small (eta^2 = 0.01 is small) .


# Cliff's delta for pairwise comparisons
delta_SM_DNF <- cliffs_delta(Steps ~ Paradigm, 
                             data = subset(df_robust_long, Paradigm %in% c("SM","DNF")))
delta_SM_BT <- cliffs_delta(Steps ~ Paradigm, 
                            data = subset(df_robust_long, Paradigm %in% c("SM","BT")))
delta_DNF_BT <- cliffs_delta(Steps ~ Paradigm,
                             data = subset(df_robust_long, Paradigm %in% c("DNF","BT")))

print(delta_SM_DNF)
print(delta_SM_BT)
print(delta_DNF_BT)

# Cohen's d is inappropriate in a non-parametric setting, which is why we omit this
# analysis in this case.
# Cliff's delta is more robust to small SDs and non-parametric data 
# than Cohen's d, hence it is an appropriate effect size measure in this case.
# We find a medium effect sizes (0.33 < delta < 0.47) for the comparisons
# between SM and DNF (delta = 0.45), as well as between DNF and BT (delta = 0.45).
# For the comparison between SM and BT (delta = 0.1), we find a negligible 
# effect size (delta < 0.147) regarding the number of steps taken until 
# the simulation concluded.

# Cliff's delta for perturbation effect
delta_perturbation <- cliffs_delta(Steps ~ Perturbation, data = df_robust_long)
print(delta_perturbation)
# We find a negligible effect size (delta < 0.147) for the effect of perturbation type
# on the number of steps per simulation (delta = 0.13).


# In conclusion we can report that there is a significant effect of paradigm choice
# on steps per simulation (p<0.001, F(2,114)=13.15, eta^2=0.19 in ART
# transformed data, large effect), per ART ANOVA analysis. 
# Moreover, the data shows perturbation type to significantly affect the same metric 
# (p = 0.01, F(1, 114) = 6.58, eta^2 = 0.05, medium effect). The interaction between
# paradigm and perturbation type was however not significant, indicating that the
# different paradigms did not react differently to the perturbations in terms of
# number of steps needed to complete a simulation (p = 0.25, F(2,114) = 1.39,
# eta^2 = 0.02, small effect).
# Post-hoc Wilcoxon tests indicate significant pairwise differences in number of steps in
# a simulation between both Behavior Trees and our Dynamic Neural Field system
# (p < 0.01, delta = 0.45, medium effect), as well as State Machines and the DNF 
# implementation (p < 0.01, delta = 0.45, medium effect). State Machines and 
# Behavior Trees did however not differ significantly from each other on this
# metric (p = 0.45, delta = 0.1, negligible effect).
# A Wilcoxon rank sum test showed that the two perturbations do not significantly 
# differ in the number of steps associated with them (W=2037.5, p=0.21, 
# delta = 0.13, negligible effect). This contradicts the finding of the ART Anova
# regarding the impact of perturbation on number of steps.
# Because of this discrepancy, I created an additional plot, visualizing the number
# of steps by perturbation type. Visual inspection and a desire to be parsimonious
# lead me to reject the finding that perturbation type significantly affects
# number of steps needed to complete a simulation.
# The total number of benchmark trials performed was 120 (20 per paradigm and perturbation).


#############
# Final Table
#############

# --- Omnibus ART ANOVA rows ---
anova_art <- anova(model_art)

anova_paradigm <- tibble(
  Comparison = "ART ANOVA (Main effect: Paradigm)",
  F_value = round(anova_art$`F value`[1], 2),
  df1 = anova_art$Df[1],
  df2 = anova_art$Df.res[1],
  p_value = signif(anova_art$`Pr(>F)`[1], 3),
  eta_sq = round(eta_squared_result$Eta2[1], 2),
  Cliffs_delta = NA
)

anova_perturbation <- tibble(
  Comparison = "ART ANOVA (Main effect: Perturbation)",
  F_value = round(anova_art$`F value`[2], 2),
  df1 = anova_art$Df[2],
  df2 = anova_art$Df.res[2],
  p_value = signif(anova_art$`Pr(>F)`[2], 3),
  eta_sq = round(eta_squared_result$Eta2[2], 2),
  Cliffs_delta = NA
)

# --- Pairwise post-hoc results (Paradigm, Wilcoxon) ---
wilcox_mat <- post_hoc_steps$p.value
wilcox_table <- as.data.frame(wilcox_mat) %>%
  rownames_to_column("Group1") %>%
  tidyr::pivot_longer(-Group1, names_to = "Group2", values_to = "p_value") %>%
  filter(!is.na(p_value)) %>%
  mutate(Comparison = paste(Group1, Group2, sep = "-"),
         p_value = signif(p_value, 3))

# --- Pairwise effect sizes (Paradigm) ---
effect_table <- tibble(
  Comparison = c("SM-BT","SM-DNF","DNF-BT"),
  Cliffs_delta = round(c(delta_SM_BT$r_rank_biserial,
                         delta_SM_DNF$r_rank_biserial,
                         delta_DNF_BT$r_rank_biserial), 2)
)

pairwise_results <- left_join(wilcox_table, effect_table, by = "Comparison")

# --- Perturbation Wilcoxon test + effect size ---
perturbation_wilcox <- wilcox.test(Steps ~ Perturbation, data = df_robust_long, exact = FALSE)
perturbation_row <- tibble(
  Comparison = "Wilcoxon (Perturbation)",
  F_value = NA,
  df1 = NA,
  df2 = NA,
  p_value = signif(perturbation_wilcox$p.value, 3),
  eta_sq = NA,
  Cliffs_delta = round(delta_perturbation$r_rank_biserial, 2)  # your effect size object
)

# --- Final combined table ---
results_table_steps <- bind_rows(anova_paradigm, anova_perturbation,
                                 pairwise_results, perturbation_row)
results_table_steps <- results_table_steps[-c(8,9)]


print(results_table_steps)


###########################################
# Number of Successes under Perturbations #
###########################################
df_robust_long$Success <- as.factor(df_robust_long$Success)

# Using binomial GLM for success data

# Assumptions:
# Binomial data - given
# Independence of observations - given

# Little/No multicollinearity between independent variables
# => should be given by design, will check anyway
# Linearity of independent variables and log odds of the dependent variable
# => also given, as all independent variables are categorical

# Checking data requirements:
table(df_robust_long$Paradigm, 
      df_robust_long$Perturbation, 
      df_robust_long$Success)

# 120 trials, 90 successes, 30 failures, no 0-cells, so we can proceed.

options(contrasts = c("contr.sum", "contr.poly"))

# Checking multicollinearity between independent variables
model_success <- glm(Success ~ Paradigm * Perturbation, 
                     data = df_robust_long, 
                     family = binomial)
vif(model_success)
# No multicollinearity, as expected

# Checking dispersion
summary(model_success)$deviance / summary(model_success)$df.residual
# 1.15 < 1.5, is fine, no overdispersion detected.

# Cook's distance to check for influential data points
cooksd <- cooks.distance(model_success)
plot(cooksd, type = "h",
     main = "Cook's Distance",
     ylab = "Cook's distance",
     xlab = "Observation index")
abline(h = 4/length(model_success$fitted.values), col = "red", lty = 2)

cooksd[cooksd > (4/length(model_success$fitted.values))]

cooksd[cooksd > 1]

# No values exceed the hard cd boundary of 1, six exceed the soft boundary of
# 4/n. Four of these are not that extreme when compared to the rest of the data,
# two, however, are, meaning they could have high influence on the model outcome.

# Leverage analysis
plot(model_success, 5)
# This does not look like there would be cause for alarm

# Performing sensitivtiy analysis to
# Identify influential points
influential <- which(cooksd > (4/length(cooksd)))

# Refit without them
model_sens <- glm(Success ~ Paradigm * Perturbation,
                  data = df_robust_long[-influential, ],
                  family = binomial)

summary(model_success)
summary(model_sens)
# There is a strong impact of the inclusion or omission of these data points
# on the model parameter estimates for the Sensor Noise perturbation level.


# Based on the above analyses, I conclude that the GLM is appropriate,
# but that certain data points have high influence on the model.


# Interpreting model
summary(model_success)
levels(df_robust_long$Paradigm)
levels(df_robust_long$Perturbation)

# The grand mean is significant (p<.001, log-odds 1.1), indicating that the overall number of
# successes is larger than chance level (50%). Moreover, we find a significant
# interaction between the DNF paradigm and the Object displacement perturbation
# (p=0.04, OR 1.85). This indicates that the DNF paradigm performs better when facing
# object displacement perturbations compared to sensor noise perturbations.
# However, the remaining results indicate that the number of successes generally
# does not differ significantly by paradigm, perturbation type, or their interaction
# besides this one exception.

# Improved version:
# The intercept, representing the grand mean under sum-to-zero contrasts, 
# was significant (p<.001, log-odds 1.1), indicating that the overall success 
# rate across all paradigms and perturbations was greater than chance (50%). 
# A significant interaction was observed between the DNF paradigm and the object 
# displacement perturbation (p=.04, OR 1.85), suggesting that DNF achieved higher 
# odds of success under object displacement compared to sensor noise. 
# No other main effects of paradigm or perturbation, nor their interactions, 
# reached statistical significance.


# Pairwise comparisons
emmeans(model_success, pairwise ~ Paradigm, adjust = "holm")
emmeans(model_success, pairwise ~ Perturbation, adjust = "holm")

# Interactions
emmeans(model_success, pairwise ~ Paradigm | Perturbation, adjust = "holm")

# Pairwise comparisons further underscore the lack of differences between the 
# various treatments and their combinations.
# No significant difference in number of successes was found between any of the 
# paradigms or the two perturbation types.

# Odds ratio - Effect Size
emmeans(model_success, pairwise ~ Paradigm, type = "response")
emmeans(model_success, pairwise ~ Perturbation, type = "response")

# Although we find that SMs have 1.5 times higher odds of success than BTs and
# 2.39 times higher odds of success than the DNF system, these values were not 
# statistically significant (p = 0.76 and p = 0.27 respectively). Similarly, 
# BTs have 1.59 times higher odds of success than the DNF system, but this
# difference was also not statistically significant (p = 0.62).
# Sensor Noise had 1.39 times higher odds of success than the Object Displacement
# perturbation, however this too was not significant (p = 0.46).

# Interactions Effect sizes
exp(coef(model_success))
pairs(emmeans(model_success, ~ Paradigm * Perturbation),
        by='Paradigm', 
        adjust='tukey')

pairs(emmeans(model_success, ~ Paradigm * Perturbation),
      by='Perturbation', 
      adjust='tukey')


# In conclusion we can report that there is no significant effect of paradigm 
# choice or perturbation applied on the number of successes in this benchmark 
# trial (all p > 0.05), as revealed by GLM (binomial family) analysis.
# This result was further confirmed by pairwise comparisons, which also showed 
# no significant differences between the three paradigms or the two perturbation types.
# Finally, odds ratio analysis showed that the trends observable in a bar plot 
# of this data were reflected in the respective odds ratios (SM > BT > DNF, 
# Sensor Noise > Object Displacement). All of these findings were however also 
# not significant.
# The total number of benchmark trials performed was 120 (20 per paradigm and perturbation).


#############
# Final Table
#############

# --- GLM significant effects ---
glm_tidy <- tidy(model_success, exponentiate = TRUE, conf.int = TRUE)

# Keep only intercept + DNF:ObjectDisplacement
glm_sig <- glm_tidy %>%
  filter(term %in% c("(Intercept)", "Paradigm2:Perturbation1")) %>%
  mutate(Interpretation = c(
    "Overall success rate significantly above chance (50%)",
    "DNF shows higher odds under Object Displacement vs Sensor Noise"
  ))

# --- emmeans pairwise comparisons ---
em <- emmeans(model_success, pairwise ~ Paradigm, expponentiate = TRUE, conf.int = TRUE)

# Pairwise comparisons paradigms (odds ratios, Tukey adjusted)
em_pairs_para <- contrast(em, method = "pairwise", adjust = "tukey", type = "response") %>%
  summary(infer=c(TRUE,TRUE)) %>%
  as.data.frame() %>%
  transmute(
    term = contrast,
    estimate = odds.ratio,
    conf.low = asymp.LCL,
    conf.high = asymp.UCL,
    p.value = p.value,
    Interpretation = ifelse(p.value < 0.05, "Significant", "Not significant")
  )


# Pairwise comparisons perturbation (odds ratios, Tukey adjusted)
em <- emmeans(model_success, pairwise ~ Perturbation, expponentiate = TRUE, conf.int = TRUE)
em_pairs_pert <- contrast(em, method = "pairwise", adjust = "tukey", type = "response") %>%
  summary(infer=c(TRUE,TRUE)) %>%
  as.data.frame() %>%
  transmute(
    term = contrast,
    estimate = odds.ratio,
    conf.low = asymp.LCL,
    conf.high = asymp.UCL,
    p.value = p.value,
    Interpretation = ifelse(p.value < 0.05, "Significant", "Not significant")
  )

# single investigative interaction 
em <- emmeans(model_success, pairwise ~ Paradigm * Perturbation, 
              expponentiate = TRUE, 
              conf.int = TRUE,
              by='Paradigm')
em_pairs_interaction <- contrast(em, method = "pairwise", adjust = "tukey", type = "response") %>%
  summary(infer=c(TRUE,TRUE)) %>%
  as.data.frame() %>%
  transmute(
    term = contrast,
    estimate = odds.ratio,
    conf.low = asymp.LCL,
    conf.high = asymp.UCL,
    p.value = p.value,
    Interpretation = ifelse(p.value < 0.05, "Significant", "Not significant")
  )



# --- Combine into one table ---
final_table <- bind_rows(
  glm_sig %>% select(term, estimate, conf.low, conf.high, p.value, Interpretation),
  em_pairs_para, em_pairs_pert, em_pairs_interaction[2,]
)

# Pretty print
final_table
# final row is specifically OD / SN for DNF
