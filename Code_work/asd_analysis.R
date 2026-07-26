# asd_analysis.R
# ISYE 7406 - Midterm: ASD Risk Screening Across Age Groups
# Abdurrahman Sabbahi | asabbahi3@gatech.edu
# -----------------------------------------------------------------------

library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(MASS)
library(pROC)
library(corrplot)
library(glmnet)
library(gridExtra)

# MASS masks dplyr -- keep the dplyr versions
select <- dplyr::select
filter <- dplyr::filter
set.seed(42)

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
script_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)),
                       error = function(e) getwd())
data_dir   <- file.path(dirname(script_dir), "ASD Different Age Group Dataset")
fig_dir    <- file.path(script_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

savefig <- function(name, w = 8, h = 5)
  ggsave(file.path(fig_dir, paste0(name, ".png")), width = w, height = h, dpi = 150)


# =======================================================================
# 1. LOAD & CLEAN
# =======================================================================

load_clean <- function(file, label) {
  df <- read.csv(file.path(data_dir, file),
                 stringsAsFactors = FALSE, na.strings = c("", "?", " "))
  names(df) <- trimws(names(df))
  names(df)[names(df) %in% c("Class.ASD", "Class/ASD")] <- "ASD"
  df %>%
    mutate(across(where(is.character), trimws),
           age_group = label,
           age       = suppressWarnings(as.numeric(age))) %>%
    select(-any_of(c("id", "age_desc", "result")))
  # age NA imputation is deferred until after the split (see Section 2)
}

child <- load_clean("Autism_Child_Data.csv",      "Child")
teen  <- load_clean("Autism_Adolescent_Data.csv", "Adolescent")
adult <- load_clean("Autism_Adult_Data.csv",       "Adult")
comb  <- bind_rows(child, teen, adult)

aq_cols <- paste0("A", 1:10, "_Score")

cat(sprintf("Loaded -- Child:%d  Adolescent:%d  Adult:%d  Combined:%d\n",
            nrow(child), nrow(teen), nrow(adult), nrow(comb)))


# =======================================================================
# 2. TRAIN / TEST SPLIT  (80/20, stratified)
#
# The split happens once and is never revisited.  Age NAs are imputed
# after the split using the training-set median so the test set cannot
# influence the imputed value.
# =======================================================================

group_list <- list(Child = child, Adolescent = teen,
                   Adult = adult, Combined = comb)

impute_age <- function(train_df, test_df) {
  train_med <- median(train_df$age, na.rm = TRUE)
  train_df$age[is.na(train_df$age)] <- train_med
  test_df$age[is.na(test_df$age)]   <- train_med
  list(train = train_df, test = test_df)
}

splits <- lapply(group_list, function(df) {
  idx <- createDataPartition(df$ASD, p = 0.8, list = FALSE)
  impute_age(df[idx, ], df[-idx, ])
})


# =======================================================================
# 3. FEATURE ENCODING
#
# Binary flags -> 0/1.
# Nominal factors (ethnicity, relation) -> dummy columns built from the
# training split only.  contry_of_res is excluded: 50+ sparse levels
# add near-zero-variance columns without measurable benefit.
# =======================================================================

NOM_COLS <- c("ethnicity", "contry_of_res", "relation")

encode_binary <- function(df) {
  df %>% mutate(
    ASD_bin         = as.integer(ASD == "YES"),
    gender          = as.integer(tolower(gender)          == "m"),
    jundice         = as.integer(tolower(jundice)         == "yes"),
    austim          = as.integer(tolower(austim)          == "yes"),
    used_app_before = as.integer(tolower(used_app_before) == "yes"),
    ethnicity       = replace_na(ethnicity,     "Unknown"),
    contry_of_res   = replace_na(contry_of_res, "Unknown"),
    relation        = replace_na(relation,      "Unknown")
  )
}

# Build dummy column schema from a training frame (first level = reference)
build_dummy_schema <- function(train_df) {
  enc <- encode_binary(train_df)
  lapply(NOM_COLS, function(col) {
    lvls <- sort(unique(enc[[col]]))
    lvls[-1]
  }) %>% setNames(NOM_COLS)
}

# Apply a schema to any frame.  Unseen test levels get all-zero columns.
apply_dummies <- function(df, schema) {
  df_enc <- encode_binary(df)
  dummy_frames <- lapply(NOM_COLS, function(col) {
    wanted <- schema[[col]]
    vals   <- df_enc[[col]]
    mm <- as.data.frame(sapply(wanted, function(lv) as.integer(vals == lv)))
    setNames(mm, gsub("[^A-Za-z0-9_]", "_", paste0(col, "_", wanted)))
  })
  df_out <- df_enc %>% select(-all_of(NOM_COLS))
  do.call(cbind, c(list(df_out), dummy_frames))
}

feat_aq <- aq_cols   # AQ-only feature set

make_Xy <- function(df, cols, schema) {
  enc  <- apply_dummies(df, schema)
  keep <- intersect(cols, names(enc))
  X    <- enc %>%
    select(all_of(keep)) %>%
    mutate(across(everything(), ~replace_na(as.numeric(.x), 0)))
  y    <- factor(enc$ASD_bin, levels = c(0, 1), labels = c("NO", "YES"))
  list(X = as.data.frame(X), y = y)
}

# Build train/test X/y pairs for every group x feature set.
# Schema is always built from that group's training data.
ds <- lapply(splits, function(sp) {
  schema     <- build_dummy_schema(sp$train)
  demo_bin   <- c("age", "gender", "jundice", "austim", "used_app_before")
  dummy_cols <- grep("^(ethnicity|relation)_",
                     names(apply_dummies(sp$train, schema)), value = TRUE)
  f_full <- c(aq_cols, intersect(c(demo_bin, dummy_cols),
                                  names(apply_dummies(sp$train, schema))))
  list(
    schema = schema,
    full   = list(train = make_Xy(sp$train, f_full, schema),
                  test  = make_Xy(sp$test,  f_full, schema)),
    aq     = list(train = make_Xy(sp$train, feat_aq, schema),
                  test  = make_Xy(sp$test,  feat_aq, schema))
  )
})

# Expose feat_full for EDA (derived from Combined, the broadest group)
feat_full <- c(aq_cols,
               intersect(c("age", "gender", "jundice", "austim", "used_app_before",
                           grep("^(ethnicity|relation)_",
                                names(apply_dummies(splits$Combined$train,
                                                    ds$Combined$schema)),
                                value = TRUE)),
                         names(ds$Combined$full$train$X)))


# =======================================================================
# 4. DESCRIPTIVE STATISTICS
# =======================================================================

desc <- map_dfr(group_list, function(df) {
  aq_sum <- rowSums(df[, aq_cols], na.rm = TRUE)
  tibble(N               = nrow(df),
         `ASD+ (%)`     = round(mean(df$ASD == "YES") * 100, 1),
         `AQ Mean`      = round(mean(aq_sum), 2),
         `AQ SD`        = round(sd(aq_sum),   2),
         `Age Mean`     = round(mean(df$age, na.rm = TRUE), 1),
         `Male (%)`     = round(mean(tolower(df$gender) == "m")    * 100, 1),
         `Jaundice (%)` = round(mean(tolower(df$jundice) == "yes") * 100, 1),
         `Fam Hx (%)`   = round(mean(tolower(df$austim)  == "yes") * 100, 1))
}, .id = "Dataset")

cat("\n-- Descriptive Statistics --\n"); print(desc)
write_csv(desc, file.path(fig_dir, "desc_stats.csv"))


# =======================================================================
# 5. EDA
# =======================================================================

# Class balance
bal <- map_dfr(group_list,
               ~.x %>% count(ASD) %>% mutate(pct = n / sum(n) * 100),
               .id = "Dataset") %>%
  mutate(Dataset = factor(Dataset, levels = c("Child", "Adolescent", "Adult", "Combined")))

ggplot(bal, aes(Dataset, pct, fill = ASD)) +
  geom_col(position = "dodge", colour = "white", width = .65) +
  scale_fill_manual(values = c(NO = "#d1d5db", YES = "#3b82d4"),
                    labels  = c(NO = "ASD-",   YES = "ASD+")) +
  labs(title = "Class Balance by Age Group", y = "% of group", x = NULL, fill = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = "top")
savefig("class_balance", 7, 4)

# AQ-10 total score distribution
aq_long <- map_dfr(list(Child = child, Adolescent = teen, Adult = adult),
                   ~.x %>% mutate(AQ = rowSums(across(all_of(aq_cols)))),
                   .id = "Dataset") %>%
  mutate(Dataset = factor(Dataset, levels = c("Child", "Adolescent", "Adult")))

ggplot(aq_long, aes(AQ, fill = ASD)) +
  geom_histogram(bins = 11, position = "identity", alpha = .65, colour = "white") +
  facet_wrap(~Dataset, scales = "free_y") +
  scale_fill_manual(values = c(NO = "#d1d5db", YES = "#3b82d4"),
                    labels  = c(NO = "ASD-",   YES = "ASD+")) +
  labs(title = "AQ-10 Total Score Distribution", x = "AQ Total", y = "Count", fill = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = "top")
savefig("aq_distribution", 11, 4)

# Correlation heatmap (Combined training split only -- no test leakage)
comb_enc <- apply_dummies(splits$Combined$train, ds$Combined$schema)
corr_mat <- comb_enc %>%
  select(all_of(c(aq_cols, "age", "gender", "jundice", "austim", "ASD_bin"))) %>%
  rename_with(~str_replace(.x, "_Score", "")) %>%
  rename(ASD = ASD_bin) %>%
  cor(use = "complete.obs")

png(file.path(fig_dir, "correlation_heatmap.png"), 1200, 1000, res = 180)
corrplot(corr_mat, method = "color", type = "lower", tl.cex = .8,
         addCoef.col = "black", number.cex = .6,
         col = colorRampPalette(c("#b91c1c", "white", "#1d4ed8"))(200),
         title = "Feature Correlation (Combined)", mar = c(0, 0, 2, 0))
dev.off()


# =======================================================================
# 6. PCA  (exploratory + used as LDA preprocessing)
#    Fit only on Combined training data.
# =======================================================================

Xtr_pca <- ds$Combined$full$train$X
pca_res  <- prcomp(scale(Xtr_pca), center = FALSE, scale. = FALSE)
var_exp  <- pca_res$sdev^2 / sum(pca_res$sdev^2)
cum_var  <- cumsum(var_exp)
cat(sprintf("\nPCA: %d PCs -> 90%% var;  %d PCs -> 95%% var\n",
            which(cum_var >= .9)[1], which(cum_var >= .95)[1]))

scree_df <- tibble(PC = seq_along(var_exp), Var = var_exp * 100, Cum = cum_var * 100)

p1 <- ggplot(scree_df %>% filter(PC <= 15), aes(PC, Var)) +
  geom_col(fill = "#3b82d4", colour = "white") +
  labs(title = "Scree", x = "PC", y = "Variance (%)") +
  scale_x_continuous(breaks = 1:15) + theme_minimal(base_size = 11)

p2 <- ggplot(scree_df, aes(PC, Cum)) +
  geom_line(colour = "#3b82d4", linewidth = .9) +
  geom_point(colour = "#3b82d4", size = 1.5) +
  geom_hline(yintercept = c(90, 95), linetype = c("dashed", "dotted"), colour = "grey50") +
  labs(title = "Cumulative Variance", x = "PCs", y = "Cum. Var (%)") +
  theme_minimal(base_size = 11)

ggsave(file.path(fig_dir, "pca_plots.png"),
       gridExtra::arrangeGrob(p1, p2, ncol = 2), width = 11, height = 4.5, dpi = 150)


# =======================================================================
# 7. CHI-SQUARE FEATURE RANKING
# =======================================================================

chi2_res <- map_dfr(c(aq_cols, "gender", "jundice", "austim"), function(col) {
  ct <- table(comb[[col]], comb$ASD)
  if (any(dim(ct) < 2)) return(NULL)
  tst <- chisq.test(ct, simulate.p.value = FALSE)
  tibble(Feature = col, Chi2 = round(tst$statistic, 2), p = round(tst$p.value, 5))
}) %>% arrange(desc(Chi2))

cat("\n-- Chi-Square Rankings --\n"); print(chi2_res)
write_csv(chi2_res, file.path(fig_dir, "chi2.csv"))

ggplot(chi2_res %>% arrange(Chi2) %>% mutate(Feature = factor(Feature, levels = Feature)),
       aes(Feature, Chi2, fill = p < .05)) +
  geom_col(colour = "white") + coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#3b82d4", "FALSE" = "#d1d5db"),
                    labels  = c("TRUE" = "p < .05", "FALSE" = "n.s.")) +
  labs(title = "Chi-Square: Feature vs ASD Outcome",
       x = NULL, y = "Chi-Square Statistic", fill = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = "top")
savefig("chi2_plot", 8, 5)


# =======================================================================
# 8. 5-FOLD CROSS-VALIDATION  (training data only)
#
# caret's preProcess arg fits the scaler inside each training fold, so
# held-out fold observations are never seen during scaling.
# LDA gets nzv + PCA reduction to avoid near-singular covariance on the
# dummy-expanded matrix.
# =======================================================================

cv_ctrl <- trainControl(method = "cv", number = 5,
                        classProbs = TRUE, summaryFunction = twoClassSummary,
                        savePredictions = "final")

model_specs <- list(
  "Logistic Reg." = list(method = "glm",  family = binomial(),
                          preProc = c("center", "scale")),
  "LDA"           = list(method = "lda",
                          preProc = c("nzv", "center", "scale", "pca")),
  "Naive Bayes"   = list(method = "nb",
                          preProc = c("nzv", "center", "scale"),
                          tuneGrid = data.frame(fL = 1, usekernel = FALSE, adjust = 1)),
  "KNN (k=5)"     = list(method = "knn",  tuneGrid = data.frame(k = 5),
                          preProc = c("center", "scale")),
  "Random Forest" = list(method = "rf",   tuneGrid = data.frame(mtry = 3),
                          preProc = c("center", "scale"))
)

run_cv <- function(ds_train, label) {
  map_dfr(names(model_specs), function(mn) {
    sp   <- model_specs[[mn]]
    args <- list(x = ds_train$X, y = ds_train$y,
                 method = sp$method, trControl = cv_ctrl, metric = "ROC",
                 preProcess = sp$preProc)
    if (!is.null(sp$family))   args$family   <- sp$family
    if (!is.null(sp$tuneGrid)) args$tuneGrid <- sp$tuneGrid

    fit <- tryCatch(do.call(train, args),
                    error = function(e) { message("skip ", mn, ": ", e$message); NULL })
    if (is.null(fit) || is.null(fit$pred)) return(NULL)

    best    <- fit$results[which.max(fit$results$ROC), ]
    p_clean <- fit$pred[!is.na(fit$pred$pred) & !is.na(fit$pred$obs), ]
    tn      <- sum(p_clean$pred == "NO"  & p_clean$obs == "NO")
    fp      <- sum(p_clean$pred == "YES" & p_clean$obs == "NO")

    tibble(Group = label, Model = mn,
           AUC  = round(best$ROC, 3),
           Acc  = round(mean(p_clean$pred == p_clean$obs), 3),
           Rec  = round(recall(p_clean$pred,    p_clean$obs, relevant = "YES"), 3),
           Prec = round(precision(p_clean$pred, p_clean$obs, relevant = "YES"), 3),
           F1   = round(F_meas(p_clean$pred,    p_clean$obs, relevant = "YES"), 3),
           Spec = if ((tn + fp) > 0) round(tn / (tn + fp), 3) else NA_real_)
  })
}

cat("\n-- Running 5-Fold CV --\n")
cv_res <- map_dfr(names(ds), function(nm) {
  cat(" ", nm, "\n")
  run_cv(ds[[nm]]$full$train, nm)
})
cv_res$Group <- factor(cv_res$Group, levels = c("Child", "Adolescent", "Adult", "Combined"))
print(cv_res %>% arrange(Group, desc(AUC)), n = 30)
write_csv(cv_res, file.path(fig_dir, "cv_results.csv"))

ggplot(cv_res, aes(reorder(Model, AUC), AUC, fill = Group)) +
  geom_col(position = "dodge", colour = "white", width = .7) + coord_flip() +
  geom_hline(yintercept = .9, linetype = "dashed", colour = "firebrick", linewidth = .6) +
  scale_fill_manual(values = c(Child = "#3b82d4", Adolescent = "#7c5cd8",
                                Adult = "#57606a", Combined   = "#1f2328")) +
  labs(title = "5-Fold CV AUC by Model & Age Group", x = NULL, y = "AUC", fill = "Group") +
  theme_minimal(base_size = 11) + theme(legend.position = "right")
savefig("cv_auc", 10, 5)


# =======================================================================
# 9. ROC CURVES  (pooled 5-fold CV, Combined training set)
# =======================================================================

Xtr   <- ds$Combined$full$train$X
ytr   <- ds$Combined$full$train$y
folds <- createFolds(ytr, k = 5, list = TRUE)

roc_list <- setNames(lapply(names(model_specs), function(mn) {
  sp  <- model_specs[[mn]]
  obs <- c(); prob <- c()
  for (idx_te in folds) {
    idx_tr <- setdiff(seq_len(nrow(Xtr)), idx_te)
    pp     <- preProcess(Xtr[idx_tr, ], method = sp$preProc)
    args   <- list(x = predict(pp, Xtr[idx_tr, ]), y = ytr[idx_tr],
                   method = sp$method,
                   trControl = trainControl(method = "none", classProbs = TRUE))
    if (!is.null(sp$family))   args$family   <- sp$family
    if (!is.null(sp$tuneGrid)) args$tuneGrid <- sp$tuneGrid
    fit <- tryCatch(do.call(train, args), error = function(e) NULL)
    if (is.null(fit)) next   # skip this fold, keep accumulating the others
    fold_prob <- predict(fit, predict(pp, Xtr[idx_te, ]), type = "prob")[, "YES"]
    fold_obs  <- as.integer(ytr[idx_te] == "YES")
    # Naive Bayes can emit NaN probabilities when a feature level is unseen in
    # a training fold.  Drop those rows rather than discarding the whole fold.
    keep <- !is.na(fold_prob) & !is.nan(fold_prob)
    prob <- c(prob, fold_prob[keep])
    obs  <- c(obs,  fold_obs[keep])
  }
  if (length(unique(obs)) < 2 || length(obs) < 10) return(NULL)
  roc(obs, prob, quiet = TRUE)
}), names(model_specs))
roc_list <- Filter(Negate(is.null), roc_list)

# Save as PNG for the Rmd to include if needed
roc_cols <- c("#3b82d4", "#7c5cd8", "#e05c23", "#16a34a", "#dc2626")
png(file.path(fig_dir, "roc_curves.png"), 750, 620, res = 110)
for (i in seq_along(roc_list)) {
  nm <- names(roc_list)[i]
  if (i == 1) plot(roc_list[[nm]], col = roc_cols[i], lty = i, lwd = 2,
                   main = "ROC - Combined (5-Fold CV)", legacy.axes = TRUE,
                   xlim = c(0, 1), ylim = c(0, 1))
  else         plot(roc_list[[nm]], col = roc_cols[i], lty = i, lwd = 2, add = TRUE)
}
legend("bottomright",
       legend = paste0(names(roc_list), " (AUC=", round(sapply(roc_list, auc), 3), ")"),
       col = roc_cols[seq_along(roc_list)], lty = seq_along(roc_list),
       lwd = 2, bty = "n", cex = .8)
dev.off()


# =======================================================================
# 10. HYPERPARAMETER TUNING  (per group, 5-fold CV)
#
# k for KNN and mtry for RF are selected independently within each
# group's own training data.  The final test-set models use these
# group-specific values, not the Combined values.
# =======================================================================

tune_group <- function(nm) {
  Xg <- ds[[nm]]$full$train$X
  yg <- ds[[nm]]$full$train$y
  knn_fit <- train(x = Xg, y = yg, method = "knn", trControl = cv_ctrl, metric = "ROC",
                   preProcess = c("center", "scale"), tuneGrid = data.frame(k = 1:15))
  rf_fit  <- train(x = Xg, y = yg, method = "rf",  trControl = cv_ctrl, metric = "ROC",
                   preProcess = c("center", "scale"), tuneGrid = data.frame(mtry = 2:8))
  list(knn      = knn_fit,
       rf       = rf_fit,
       best_k   = knn_fit$bestTune$k,
       best_mtry= rf_fit$bestTune$mtry,
       knn_auc  = round(max(knn_fit$results$ROC), 3),
       rf_auc   = round(max(rf_fit$results$ROC),  3))
}

cat("\n-- Hyperparameter Tuning (per group) --\n")
tune_res <- setNames(lapply(names(ds), function(nm) {
  cat(" ", nm, "\n"); tune_group(nm)
}), names(ds))

for (nm in names(tune_res)) {
  cat(sprintf("  %-12s  KNN k=%-2d (AUC=%.3f)  RF mtry=%-2d (AUC=%.3f)\n",
              nm,
              tune_res[[nm]]$best_k,    tune_res[[nm]]$knn_auc,
              tune_res[[nm]]$best_mtry, tune_res[[nm]]$rf_auc))
}

# Combined scalars used in the glmnet section and inline report text
Xc           <- ds$Combined$full$train$X
yc           <- ds$Combined$full$train$y
knn_tune     <- tune_res$Combined$knn
rf_tune      <- tune_res$Combined$rf
knn_best_k   <- tune_res$Combined$best_k
knn_best_auc <- tune_res$Combined$knn_auc
rf_best_mtry <- tune_res$Combined$best_mtry
rf_best_auc  <- tune_res$Combined$rf_auc

pk <- ggplot(knn_tune$results, aes(k, ROC)) +
  geom_line(colour = "#3b82d4", linewidth = .9) +
  geom_point(colour = "#3b82d4", size = 2) +
  geom_vline(xintercept = knn_best_k, linetype = "dashed", colour = "firebrick") +
  labs(title = sprintf("KNN: best k=%d (AUC=%.3f)", knn_best_k, knn_best_auc),
       x = "k", y = "CV AUC") +
  scale_x_continuous(breaks = 1:15) + theme_minimal(base_size = 11)

pr <- ggplot(rf_tune$results, aes(mtry, ROC)) +
  geom_line(colour = "#3b82d4", linewidth = .9) +
  geom_point(colour = "#3b82d4", size = 2) +
  geom_vline(xintercept = rf_best_mtry, linetype = "dashed", colour = "firebrick") +
  labs(title = sprintf("RF: best mtry=%d (AUC=%.3f)", rf_best_mtry, rf_best_auc),
       x = "mtry", y = "CV AUC") +
  scale_x_continuous(breaks = 2:8) + theme_minimal(base_size = 11)

ggsave(file.path(fig_dir, "tuning.png"),
       gridExtra::arrangeGrob(pk, pr, ncol = 2), width = 12, height = 4.5, dpi = 150)


# =======================================================================
# 11. REGULARIZED LOGISTIC REGRESSION
#
# Unregularized LR nearly tops out the AUC scale.  A glmnet sweep over
# alpha and lambda tests whether that reflects genuine near-separability
# or unstable coefficient estimates.
# =======================================================================

glmnet_grid <- expand.grid(alpha  = c(0, .5, 1),
                            lambda = 10^seq(-4, 0, length.out = 30))

glmnet_fit  <- train(x = Xc, y = yc, method = "glmnet", trControl = cv_ctrl,
                     metric = "ROC", preProcess = c("center", "scale"),
                     tuneGrid = glmnet_grid)
glmnet_best <- glmnet_fit$bestTune
glmnet_auc  <- round(max(glmnet_fit$results$ROC), 3)
lr_unreg    <- cv_res %>% filter(Group == "Combined", Model == "Logistic Reg.") %>% pull(AUC)

cat(sprintf("\nglmnet: alpha=%.1f  lambda=%.5f  AUC=%.3f  (unregularized=%.3f)\n",
            glmnet_best$alpha, glmnet_best$lambda, glmnet_auc, lr_unreg))

ggplot(glmnet_fit$results %>% filter(alpha == glmnet_best$alpha) %>% arrange(lambda),
       aes(log10(lambda), ROC)) +
  geom_line(colour = "#3b82d4", linewidth = .9) +
  geom_point(colour = "#3b82d4", size = 1.8) +
  geom_vline(xintercept = log10(glmnet_best$lambda), linetype = "dashed",
             colour = "firebrick") +
  labs(title = sprintf("Regularized LR penalty path  [alpha=%.1f]", glmnet_best$alpha),
       x = "log10(lambda)", y = "CV AUC") +
  theme_minimal(base_size = 11)
savefig("glmnet_path", 8, 4.5)


# =======================================================================
# 12. RF VARIABLE IMPORTANCE  (Combined training set)
# =======================================================================

pp_sc  <- preProcess(Xc, method = c("center", "scale"))
rf_imp <- randomForest(x = predict(pp_sc, Xc), y = yc, ntree = 500, importance = TRUE)

imp_df <- importance(rf_imp, type = 1) %>%
  as.data.frame() %>%
  rownames_to_column("Feature") %>%
  rename(MDA = MeanDecreaseAccuracy) %>%
  arrange(desc(MDA))

cat("\n-- Top 10 Features (RF Importance) --\n"); print(head(imp_df, 10))

ggplot(imp_df %>% slice_max(MDA, n = 15) %>% arrange(MDA) %>%
         mutate(Feature = factor(Feature, levels = Feature)),
       aes(Feature, MDA)) +
  geom_col(fill = "#3b82d4", colour = "white") + coord_flip() +
  labs(title = "RF Variable Importance (top 15) - Combined Training Set",
       x = NULL, y = "Mean Decrease in Accuracy") +
  theme_minimal(base_size = 11)
savefig("rf_importance", 8, 6)


# =======================================================================
# 13. FULL vs AQ-ONLY FEATURE SET COMPARISON
# =======================================================================

cat("\n-- Full vs AQ-Only CV --\n")
compare_res <- map_dfr(names(ds), function(nm) {
  cat(" ", nm, "\n")
  bind_rows(
    run_cv(ds[[nm]]$full$train, nm) %>% mutate(FeatSet = "Full"),
    run_cv(ds[[nm]]$aq$train,   nm) %>% mutate(FeatSet = "AQ-Only")
  )
})
compare_res$Group <- factor(compare_res$Group,
                             levels = c("Child", "Adolescent", "Adult", "Combined"))
write_csv(compare_res, file.path(fig_dir, "feature_comparison.csv"))

comp_comb <- compare_res %>%
  filter(Group == "Combined") %>%
  mutate(FeatSet = factor(FeatSet, levels = c("Full", "AQ-Only")))

ggplot(comp_comb, aes(reorder(Model, AUC), AUC, fill = FeatSet)) +
  geom_col(position = "dodge", colour = "white", width = .65) + coord_flip() +
  geom_hline(yintercept = .9, linetype = "dashed", colour = "firebrick", linewidth = .5) +
  scale_fill_manual(values = c(Full = "#3b82d4", "AQ-Only" = "#d1d5db")) +
  labs(title = "Full vs AQ-Only Feature Set (Combined, CV AUC)",
       x = NULL, y = "AUC", fill = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = "top")
savefig("feature_compare", 9, 5)


# =======================================================================
# 14. FINAL TEST SET EVALUATION
#
# Refit each model on the full training set with its group-specific tuned
# hyperparameters, then evaluate once on the held-out 20%.
# =======================================================================

eval_on_test <- function(nm) {
  tr <- ds[[nm]]$full$train
  te <- ds[[nm]]$full$test

  specs <- list(
    "Logistic Reg." = list(method = "glm",  family = binomial(),
                            preProc = c("center", "scale"),     tuneGrid = NULL),
    "LDA"           = list(method = "lda",
                            preProc = c("nzv", "center", "scale", "pca"), tuneGrid = NULL),
    "KNN (tuned)"   = list(method = "knn",
                            preProc = c("center", "scale"),
                            tuneGrid = data.frame(k = tune_res[[nm]]$best_k)),
    "Random Forest" = list(method = "rf",
                            preProc = c("center", "scale"),
                            tuneGrid = data.frame(mtry = tune_res[[nm]]$best_mtry))
  )

  map_dfr(names(specs), function(mn) {
    sp     <- specs[[mn]]
    pp_fit <- preProcess(tr$X, method = sp$preProc)
    Xtr_pp <- predict(pp_fit, tr$X)
    Xte_pp <- predict(pp_fit, te$X)

    args <- list(x = Xtr_pp, y = tr$y, method = sp$method,
                 trControl = trainControl(method = "none", classProbs = TRUE))
    if (!is.null(sp$family))   args$family   <- sp$family
    if (!is.null(sp$tuneGrid)) args$tuneGrid <- sp$tuneGrid

    fit <- tryCatch(do.call(train, args), error = function(e) NULL)
    if (is.null(fit)) return(NULL)

    pc  <- predict(fit, Xte_pp)
    pp2 <- predict(fit, Xte_pp, type = "prob")[, "YES"]
    tibble(Group = nm, Model = mn,
           AUC  = round(auc(roc(as.integer(te$y == "YES"), pp2, quiet = TRUE)), 3),
           Acc  = round(mean(pc == te$y), 3),
           Rec  = round(recall(pc,    te$y, relevant = "YES"), 3),
           Prec = round(precision(pc, te$y, relevant = "YES"), 3),
           F1   = round(F_meas(pc,    te$y, relevant = "YES"), 3))
  })
}

cat("\n-- Final Test Set Results --\n")
test_res <- map_dfr(names(ds), eval_on_test)
test_res$Group <- factor(test_res$Group, levels = c("Child", "Adolescent", "Adult", "Combined"))
print(test_res %>% arrange(Group, desc(AUC)), n = 30)
write_csv(test_res, file.path(fig_dir, "test_results.csv"))


# =======================================================================
# 15. CONFUSION MATRICES  (RF with each group's tuned mtry)
# =======================================================================

png(file.path(fig_dir, "confusion_matrices.png"), 1100, 300, res = 110)
par(mfrow = c(1, 4), mar = c(3, 3, 3, 1))
for (nm in names(ds)) {
  tr   <- ds[[nm]]$full$train
  te   <- ds[[nm]]$full$test
  pp   <- preProcess(tr$X, method = c("center", "scale"))
  mtry <- tune_res[[nm]]$best_mtry     # group-specific tuned value
  fit  <- randomForest(x = predict(pp, tr$X), y = tr$y, ntree = 200, mtry = mtry)
  pred <- predict(fit, predict(pp, te$X))
  cm   <- prop.table(table(Pred = pred, True = te$y))
  corrplot(cm, is.corr = FALSE, method = "color", tl.cex = .8,
           col = colorRampPalette(c("#dbeafe", "#1d4ed8"))(50),
           addCoef.col = "black", number.cex = .9,
           title = paste0(nm, " (Acc=", round(mean(pred == te$y), 3), ")"),
           mar = c(0, 0, 2, 0))
}
dev.off()


# =======================================================================
# 16. SUMMARY
# =======================================================================

cat("\n== Best Model per Group (Test AUC) ==\n")
test_res %>%
  group_by(Group) %>%
  slice_max(AUC, n = 1) %>%
  select(Group, Model, AUC, Acc, Rec, F1) %>%
  print()

cat(sprintf("\nglmnet vs unregularized LR (Combined CV AUC): %.3f vs %.3f\n",
            glmnet_auc, lr_unreg))
cat(sprintf("Best KNN k=%d (AUC=%.3f)   Best RF mtry=%d (AUC=%.3f)\n",
            knn_best_k, knn_best_auc, rf_best_mtry, rf_best_auc))
cat("\nDone. Figures -> Code_work/figures/\n")
