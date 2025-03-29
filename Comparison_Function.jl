
# Main function
function Comparison_Function(N, P, k, random_true_columns, SNR, rho, Nsim)

    random_true_columns = random_index_true_columns
    SNR =  SNR[1]
    # Generate data
    result = Data_Generation(N, P, k, random_true_columns, SNR, rho)
    X = result[:X] 
    Y = result[:Y]
    true_beta = result[:true_beta] 
    X_covariance = result[:cov] 
    Y_variance = result[:variance]
    
    
    # Random initial columns
    random_initial_columns = sort(sample(1:P, 30, replace=false))

    # Perform variable selection with GIC
    AIC_result =  GIC_Variable_Selection(X, Y, random_initial_columns, Calculate_AIC, Calculate_AIC_short)
    BIC_result =  GIC_Variable_Selection(X, Y, random_initial_columns, Calculate_BIC, Calculate_BIC_short)
    SIC_result =  GIC_Variable_Selection(X, Y, random_initial_columns, Calculate_SIC, Calculate_SIC_short)
    CAIC_result =  GIC_Variable_Selection(X, Y, random_initial_columns, Calculate_CAIC, Calculate_CAIC_short)
    CAICF_result =  GIC_Variable_Selection(X, Y, random_initial_columns, Calculate_CAICF, Calculate_CAICF_short)


    glmnet_Variable_Selection(X, Y, relax_indicator=false, adaptive_indicator=false)

    # Perform variable selection lasso and other adaptions
    lasso = glmnet_Variable_Selection(X, Y, false, false)
    relax_lasso = glmnet_Variable_Selection(X, Y, true, false)
    adaptive_lasso = glmnet_Variable_Selection(X, Y, false, true)

    # Evaluation metrics
    AIC_EM = Eval_metric(X, Y, AIC_result[2], X_covariance, true_beta, Y_variance)
    BIC_EM = Eval_metric(X, Y, BIC_result[2], X_covariance, true_beta, Y_variance)
    CAICF_EM = Eval_metric(X, Y, CAICF_result[2], X_covariance, true_beta, Y_variance)
    Lasso_EM = Eval_metric(X, Y, 0, X_covariance, true_beta, Y_variance, beta_hat=lasso, Lasso=true)
    relax_Lasso_EM = Eval_metric(X, Y, 0, X_covariance, true_beta, Y_variance, beta_hat=relax_lasso, Lasso=true)
    adaptive_Lasso_EM = Eval_metric(X, Y, 0, X_covariance, true_beta, Y_variance, beta_hat=adaptive_lasso, Lasso=true)

    # Relative risks, test errors, variance explained
    RR_o = [AIC_EM[1], BIC_EM[1], CAICF_EM[1], Lasso_EM[1], relax_Lasso_EM[1], adaptive_Lasso_EM[1]]
    RR_i = [AIC_EM[2], BIC_EM[2], CAICF_EM[2], Lasso_EM[2], relax_Lasso_EM[2], adaptive_Lasso_EM[2]]
    RTE = [AIC_EM[3], BIC_EM[3], CAICF_EM[3], Lasso_EM[3], relax_Lasso_EM[3], adaptive_Lasso_EM[3]]
    PVE = [AIC_EM[4], BIC_EM[4], CAICF_EM[4], Lasso_EM[4], relax_Lasso_EM[4], adaptive_Lasso_EM[4]]

    # Overselection analysis
    VS_AIC = Overselection(AIC_result[2], random_true_columns)
    VS_BIC = Overselection(BIC_result[2], random_true_columns)
    VS_CAICF = Overselection(CAICF_result[2], random_true_columns)
    VS_lasso = Overselection(lasso, random_true_columns)
    VS_relax_lasso = Overselection(relax_lasso, random_true_columns)
    VS_adaptive_lasso = Overselection(adaptive_lasso, random_true_columns)

    # Combine results
    VS_selection = vcat(VS_AIC, VS_BIC, VS_CAICF, VS_lasso, VS_relax_lasso, VS_adaptive_lasso)
    label = ["AIC", "BIC", "CAICF", "Lasso", "Relax_Lasso", "Adaptive_Lasso"]

    return DataFrame(
        RR_o = RR_o,
        RR_i = RR_i,
        RTE = RTE,
        PVE = PVE,
        Over = VS_selection[:, 1],
        Under = VS_selection[:, 2],
        Correct = VS_selection[:, 3],
        Label = label,
        SNR = SNR,
        nsim = Nsim
    )
end