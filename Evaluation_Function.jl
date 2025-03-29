function Eval_metric(X, Y, COV, beta_t, variance; 
    IC_Selected_Coeff=nothing, beta_hat=nothing, Lasso=false)

# Input:
# X: Design matrix
# Y: Outcome vector
# COV: Covariance matrix
# beta_t: True coefficients
# variance: Variance of Y
# IC_Selected_Coeff: Indices of selected variables (default: nothing)
# beta_hat: Estimated coefficients (default: nothing)
# Lasso: Boolean indicating whether Lasso coefficients are used (default: false)

P = size(X, 2)  # Number of predictors

# Initialize beta_hat
if !Lasso
if IC_Selected_Coeff === nothing || isempty(IC_Selected_Coeff)
beta_hat = zeros(P)
else
# Subset X based on selected coefficients
X_ss = X[:, IC_Selected_Coeff]

# Estimate beta_hat for the selected coefficients
beta_hat = zeros(P)
beta_hat[IC_Selected_Coeff] = X_ss \ Y
end
end

# Compute metrics
diff = beta_hat - beta_t
true_term = beta_t' * COV * beta_t

# Relative Risk Out-of-Sample
RR_o = (diff' * COV * diff) / true_term

# Relative Risk In-Sample
denom_in_sample = beta_hat' * COV * beta_hat
RR_i = ifelse(denom_in_sample == 0, NaN, (diff' * COV * diff) / denom_in_sample)

# Relative Test Error
RTE = (diff' * COV * diff + variance) / variance

# Proportion of Variance Explained
PVE = 1 - (diff' * COV * diff + variance) / (true_term + variance)

return [RR_o, RR_i, RTE, PVE]
end
