# Multiple Training & Testing Sample Method (MuTaTe)

setwd("~/Documents/R Files/PRED422 - Machine Learning Class")
library(caret)
library(dplyr)
data("GermanCredit") # dataset in caret
credit <- data.frame(GermanCredit) # target = Amount


# Exploratory Analysis for predictors ##############################################################

# check for missing values
na.counter.df <- function(df) {
  
  columns <- ncol(df)
  total_obs <- nrow(df)
  
  var <- c() # initialize variable names vector
  missing <- numeric() #initialize
  propn_missing <- numeric() #initialize
  
  for (i in 1:columns) {
    
    var[i] <- colnames(df[i])
    missing[i] <- sum(is.na(df[i]))
    propn_missing[i] <- round(missing[i] / total_obs, digits = 3)
    
  }
  
  dframe <- data.frame(var, missing, propn_missing)
  dframe
}

na.counter.df(credit) # no missing values in any variables
dim(credit)
glimpse(credit)


# use rpart to explore variable importance. 
library(rpart)
set.seed(82510)
train.tree.index <- sample(nrow(credit), size = round(0.70 * nrow(credit)))
credit.train <- credit[train.tree.index, ]
credit.holdout <- credit[-train.tree.index, ] 
tree <- rpart(Amount ~., data = credit.train, method = "anova")
summary(tree)

tree.pred <- predict(tree, newdata = credit.holdout)
# Calculate R-Squared on holdout - The square of correlation b/w actual and predicted holdout values
cor(credit.holdout$Amount, tree.pred)^2 # 0.46 R-Squared in holdout
sqrt(mean((credit.holdout$Amount - tree.pred)^2)) # residual standard error holdout

# extracted predictor importance values from rpart output.scaled to be proportion of 1
pred.importance <- round(tree$variable.importance / sum(tree$variable.importance), 2)
pred.importance
pred.importance[1:10]


# Use LASSO regression to identify best predictor variable subset
library(glmnet)
# lambda tuning parameter
grid <- 10 ^ seq(from = 10, to = -2, length = 100) 
# x matrix, y vector
x <- model.matrix(Amount~., data = credit)[, -2]
y <- credit$Amount
# train and test indices
set.seed(82510)
train <- sample(nrow(x), size =  round(0.70 * nrow(x)))
y.test <- y[-train]
# automatically standardizes variables. alpha argument is to ensure LASSO method
# use 10-fold cv on the training dataset. ISLR pp 253-254.
set.seed(1) # reproducible folds
lasso.cv <- cv.glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
plot(lasso.cv)
bestlam <- lasso.cv$lambda.min
bestlam
lasso.cv$nzero
lasso.cv$glmnet.fit

# predict Mean Squared Error on test data using the bestlam tuning parameter model.
lasso.pred <- predict(object = lasso.cv, s = bestlam, newx = x[-train, ])
mean((y.test - lasso.pred)^2) # MSE
sqrt(mean((y.test - lasso.pred)^2)) # residual standard error
cor(y.test, lasso.pred)^2 # r2holdout 

# refit the full model using best lambda with lowest cv mean squared error
full.lasso <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(full.lasso, type ="coefficients", s = bestlam)[1:37,] # 37 nonzero coef
lasso.coef[lasso.coef != 0]
sort(abs(lasso.coef[lasso.coef != 0]), decreasing = T)


# Re-Sampling Method - Multiple Train & Test (MuTaTe) ##############################################

# credit dataset in caret package
# initiate empty vectors to collect values
intercept <- c()
Duration <- c()
Job.Management.SelfEmp.HighlyQualified <- c()
InstallmentRatePercentage <- c()
Purpose.UsedCar <- c()
rsquared.train <- c()
rsquared.holdout <- c()  

# create 1000 separate OLS fits using different 90/10 training/validation sets.
for (i in 1: 1000) {
  
  # create reproducible, randomly sampled training vs validation data.
  set.seed(i + 1)
  indices <- sample(nrow(credit), size = round(0.90 * nrow(credit)))
  data.train <- credit[indices, ]
  data.holdout <- credit[-indices, ] 
  
  # linear regression model fit with training data. Predictors chosen from prior EDA models.
  model.fit <- lm(Amount ~ Duration + Job.Management.SelfEmp.HighlyQualified + 
                 InstallmentRatePercentage + Purpose.UsedCar , data = data.train)
  
  # capture fitted coefficient values.
  coeff <- coefficients(model.fit)
  
  intercept[i] <- coeff[1]
  Duration[i] <- coeff[2]
  Job.Management.SelfEmp.HighlyQualified[i] <- coeff[3]
  InstallmentRatePercentage[i] <- coeff[4]
  Purpose.UsedCar[i] <- coeff[5]
  
  # capture r-squared from training
  rsquared.train[i] <- summary(model.fit)$r.squared
  
  # predict model.fit on unseen records, store predictions.
  predicted.amount <- predict(model.fit, newdata = data.holdout)
  # calculate the r-squared on holdout data.
  # square the correlation between actual vs predicted amount.
  rsquared.holdout[i] <- cor(data.holdout$Amount, predicted.amount)^2
  
}   

# tidy results into a data frame.
mutate.results <- data.frame(model_number = rep(1:1000), rsquared.train, rsquared.holdout,intercept, 
                             Duration, Job.Management.SelfEmp.HighlyQualified, 
                             InstallmentRatePercentage, Purpose.UsedCar)

# dplyr to generate summary stats about sampling distributions.
mutate.stats <- mutate.results %>%
  summarise(intercept.mean.coef = mean(intercept),
            intercept.sd.coef = sd(intercept),
            Duration.mean.coef = mean(Duration),
            Duration.sd.coef = sd(Duration),
            Job.Management.SelfEmp.HighlyQualified.mean.coef = mean(Job.Management.SelfEmp.HighlyQualified),
            Job.Management.SelfEmp.HighlyQualified.sd.coef = sd(Job.Management.SelfEmp.HighlyQualified),
            InstallmentRatePercentage.mean.coef = mean(InstallmentRatePercentage),
            InstallmentRatePercentage.sd.coef = sd(InstallmentRatePercentage),
            Purpose.UsedCar.mean.coef = mean(Purpose.UsedCar),
            Purpose.UsedCar.sd.coef = sd(Purpose.UsedCar),
            rsquared.train.mean = mean(rsquared.train),
            rsquared.train.sd = sd(rsquared.train),
            rsquared.holdout.mean = mean(rsquared.holdout),
            rsquared.holdout.sd = sd(rsquared.holdout),
            rsquared.difference.mean = mean(rsquared.holdout.mean - rsquared.train.mean)
            )

coef.means <- rbind(mutate.stats[, c(1,3,5,7,9)])
coef.sd <- rbind(mutate.stats[, c(2,4,6,8,10)])

reshape.coef.stats <- data.frame(coefficient = c("intercept", "Duration", 
                                           "Job.Management.SelfEmp.HighlyQualified", 
                                           "InstallmentRatePercentage",
                                           "Purpose.UsedCar"),
                           coef.means = as.numeric(coef.means), coef.sd = as.numeric(coef.sd))
reshape.coef.stats

# graph out the results ############################################################################

par(mfrow = c(3,2))

# distribution of intercept.
hist(mutate.results$intercept, col = "grey", xlab = "Intercept",
     main = paste("Coefficient Distribution of", "Intercept"))
abline(v = mutate.stats$intercept.mean.coef, lty = "dashed", lwd = "3", col = "blue")  

# distribution of duration predictor coef.
hist(mutate.results$Duration, col = "grey", main = paste("Coefficient Distribution of", "Duration"),
     xlab = "Duration")
abline(v = mutate.stats$Duration.mean.coef, lty = "dashed", lwd = "3", col = "blue")  

# distribution of Job.Management.SelfEmp.HighlyQualified predictor coef.
hist(mutate.results$Job.Management.SelfEmp.HighlyQualified, col = "grey",
     main = paste("Coefficient Distribution of", "Job.Management.SelfEmp.HighlyQualified"),
     xlab = "Job.Management.SelfEmp.HighlyQualified")
abline(v = mutate.stats$Job.Management.SelfEmp.HighlyQualified.mean.coef, 
       lty = "dashed", lwd = "3", col = "blue")  

# distribution of InstallmentRatePercentage predictor coef.  
hist(mutate.results$InstallmentRatePercentage, col = "grey",
     main = paste("Coefficient Distribution of", "InstallmentRatePercentage"),
     xlab = "InstallmentRatePercentage")
abline(v = mutate.stats$InstallmentRatePercentage.mean.coef, lty = "dashed", lwd = "3", 
       col = "blue")  

# distribution of Purpose.UsedCar  predictor coef. 
hist(mutate.results$Purpose.UsedCar, col = "grey",
     main = paste("Coefficient Distribution of", "Purpose.UsedCar"), xlab = "Purpose.UsedCar")
abline(v = mutate.stats$Purpose.UsedCar.mean.coef, lty = "dashed", lwd = "3", 
       col = "blue") 

par(mfrow = c(2,1))
# distribution of holdout Rsquared.
hist(mutate.results$rsquared.holdout, xlab = "R-Squared Holdout", col = "grey",
     main = "Distribution of R-Squared values from Holdout Data")
abline(v = mutate.stats$rsquared.holdout.mean, lty = "dashed", lwd = "3", col = "blue")  


# distribution of % Decrease in holdout Rsquared distribution.
hist(mutate.results$rsquared.holdout - mutate.results$rsquared.train, xlab = "R-Squared % Decrease",
     col = "grey", main = "Distribution of R-Squared % Decrease in Holdout")
abline(v = mutate.stats$rsquared.difference.mean, lty = "dashed", lwd = "3", col = "blue")  


# Compare these resampling results to a single model built on entire sample #######################

full.lmfit <- lm(Amount ~ Duration + Job.Management.SelfEmp.HighlyQualified + 
                   InstallmentRatePercentage + Purpose.UsedCar, data = credit)
summary(full.lmfit)

# names of different objects saved with the fitted model.
names(full.lmfit) 

# names of objects within the summary function of the fitted model
names(summary(lm.fit)) 

# save the model coefficients
full.lmfit.coef <- data.frame(coefficients(lm.fit))
full.lmfit.coef
round(summary(full.lmfit)$r.squared, 3) # full model Rsqrd


