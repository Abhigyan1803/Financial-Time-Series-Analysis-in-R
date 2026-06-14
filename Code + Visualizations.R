install.packages(c("quantmod", "forecast", "tidyverse", "lubridate", "ranger", "caret", "xgboost", "h2o", "corrplot", "reshape2"))
library(h2o)
h2o.init()
library(quantmod)
library(forecast)
library(tidyverse)
library(lubridate)
library(ranger)
library(caret)
library(xgboost)
library(h2o)
library(corrplot)
library(reshape2)

set.seed(123)
h2o.init(nthreads = -1, max_mem_size = "4G")

# Fetch Treasury yield data from FRED
get_treasury_data <- function(start_date = "2010-01-01", end_date = Sys.Date()) {
  treasury_tickers <- c("DGS1", "DGS3", "DGS5", "DGS10")
  data_list <- lapply(treasury_tickers, function(ticker) {
    tryCatch({
      getSymbols(ticker, src = "FRED", from = start_date, to = end_date, auto.assign = FALSE)
    }, error = function(e) NULL)
  })
  data <- do.call(merge, data_list)
  data <- na.locf(data) |> na.omit()
  return(data.frame(date = index(data), coredata(data)))
}

# Feature engineering
create_features <- function(data, lag_days = 30) {
  feature_names <- colnames(data)[-1]
  for (col in feature_names) {
    for (lag in 1:lag_days) {
      data[[paste0(col, "_lag", lag)]] <- lag(data[[col]], lag)
    }
    data[[paste0(col, "_diff1")]] <- c(NA, diff(data[[col]]))
  }
  return(na.omit(data))
}

# Prepare model data
prepare_model_data <- function(data, target_col, horizon = 1) {
  target <- lead(data[[target_col]], horizon)
  valid_idx <- which(!is.na(target))
  return(list(
    predictors = data[valid_idx, -1],
    target = target[valid_idx],
    current = data[valid_idx - 1, target_col]
  ))
}

# Train and evaluate models
train_models <- function(predictors, target) {
  dtrain <- xgb.DMatrix(data = as.matrix(predictors), label = target)
  params <- list(objective = "reg:squarederror", eta = 0.03, max_depth = 6)
  xgb_model <- xgb.train(params, dtrain, nrounds = 500, verbose = 0)
  
  h2o_data <- as.h2o(cbind(predictors, target))
  h2o_gbm <- h2o.gbm(y = "target", training_frame = h2o_data, ntrees = 500, max_depth = 6)
  
  return(list(xgb = xgb_model, h2o = h2o_gbm))
}

# Evaluate model
evaluate <- function(actual, predicted, current) {
  valid_idx <- which(!is.na(current))
  actual <- actual[valid_idx]
  predicted <- predicted[valid_idx]
  current <- current[valid_idx]
  
  actual_dir <- actual > current
  pred_dir <- predicted > current
  dir_accuracy <- mean(actual_dir == pred_dir, na.rm = TRUE) * 100
  return(dir_accuracy)
}

# Main process
main <- function() {
  data <- get_treasury_data()
  data <- create_features(data)
  results <- list()
  predictions_all <- list()
  
  for (yield_col in colnames(data)[2:5]) {
    model_data <- prepare_model_data(data, yield_col)
    models <- train_models(model_data$predictors, model_data$target)
    
    xgb_preds <- predict(models$xgb, as.matrix(model_data$predictors))
    h2o_preds <- as.vector(h2o.predict(models$h2o, as.h2o(model_data$predictors))$predict)
    
    ensemble_preds <- (xgb_preds + h2o_preds) / 2
    accuracy <- evaluate(model_data$target, ensemble_preds, model_data$current)
    results[[yield_col]] <- accuracy
    
    predictions_all[[yield_col]] <- data.frame(
      date = data$date[(nrow(data) - length(ensemble_preds) + 1):nrow(data)],
      actual = model_data$target,
      predicted = ensemble_preds
    )
    
    cat(sprintf("%s Directional Accuracy: %.2f%%\n", yield_col, accuracy))
  }
  
  cat("Average Accuracy:", mean(unlist(results)), "%\n")
  
  # ---- VISUALIZATIONS ----

  # ---- 1. Yield Curves Over Time ----
  yield_data <- data[, c("date", "DGS1", "DGS3", "DGS5", "DGS10")]
  yield_long <- pivot_longer(yield_data, -date, names_to = "Maturity", values_to = "Yield")
  
  p1 <- ggplot(yield_long, aes(x = date, y = Yield, color = Maturity)) +
    geom_line() +
    labs(title = "Treasury Yields Over Time", x = "Date", y = "Yield (%)")
  print(p1)
  
  # ---- 2. First Differences (DGS10) ----
  p2 <- ggplot(data, aes(x = date, y = DGS10_diff1)) +
    geom_line(color = "tomato") +
    labs(title = "First Difference of DGS10", x = "Date", y = "DGS10_diff1")
  print(p2)
  
  # ---- 3. Directional Accuracy Bar Plot ----
  acc_df <- data.frame(Maturity = names(results), Accuracy = unlist(results))
  p3 <- ggplot(acc_df, aes(x = Maturity, y = Accuracy, fill = Maturity)) +
    geom_bar(stat = "identity") +
    labs(title = "Directional Accuracy by Yield", y = "Accuracy (%)") +
    theme_minimal()
  print(p3)
  
  # ---- 4. Actual vs Predicted (DGS10 example) ----
  pred_df <- predictions_all[["DGS10"]]
  pred_df$Diff <- pred_df$actual - pred_df$predicted
  
  p4 <- ggplot(pred_df, aes(x = date)) +
    geom_line(aes(y = actual), color = "blue", size = 1, alpha = 0.7) +
    geom_line(aes(y = predicted), color = "red", size = 1, alpha = 0.7) +
    labs(title = "Actual vs Predicted Yield (DGS10)", x = "Date", y = "Yield")
  print(p4)
  
  # ---- 5. Correlation Heatmap ----
  numeric_data <- select(data, where(is.numeric))
  corr_matrix <- cor(numeric_data, use = "complete.obs")
  
  corrplot(corr_matrix[1:10, 1:10], method = "color", type = "lower",
           title = "Correlation of Top Features", mar = c(0,0,2,0))

}

# Run
main()
