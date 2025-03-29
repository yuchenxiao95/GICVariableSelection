using GLMNet, Plots, LinearAlgebra, LaTeXStrings

# There is currently no implementation of relax lasso in julia, so elastic net is used instead 
function glmnet_Variable_Selection(X, Y; type = "lasso")
    
    if type == "lasso"
        # Standard Lasso or Relaxed Lasso
        lasso_fit = glmnet(X, Y, alpha=1.0)  # Perform Lasso
        cvfit = glmnetcv(X, Y, alpha = 1.0)  # Cross-validation
        
    elseif type == "ridge"
        # Perform Ridge regression for adaptive Lasso weights
        ridge_fit = glmnet(X, Y, alpha=0.0)  # Ridge regression
        cvfit = glmnetcv(X, Y)  # Cross-validation for Ridge

    elseif type == "adaptive" 

        # Perform Ridge regression for adaptive Lasso weights
        ridge_fit = glmnet(X, Y, alpha=0.0)  # Ridge regression
        ridge_cvfit = glmnetcv(X, Y)  # Cross-validation for Ridge
        # Extract Ridge coefficients (exclude intercept)
        best_lambda_index = argmin(ridge_cvfit.lambda)  # Find index of optimal lambda
        best_ridge_coef = ridge_fit.betas[:, best_lambda_index]  # Coefficients for optimal lambda
        
        # Perform adaptive Lasso with penalty factors based on Ridge coefficients
        penalty_factor = 1.0 ./ abs.(best_ridge_coef .+ 1e-5)  # Compute penalty factors

        adaptive_fit = glmnet(X, Y, alpha=1.0, penalty_factor=penalty_factor)
        cvfit = glmnetcv(X, Y, alpha = 1.0, penalty_factor=penalty_factor)  # Cross-validation for adaptive Lasso
    end

    # Extract coefficients at optimal lambda
    optimal_lambda_index = argmin(cvfit.meanloss)  # Index of optimal lambda
    lasso_coefficients = lasso_fit[:, optimal_lambda_index]

    best_lambda_index = argmin(cvfit.meanloss)  # Find index of optimal lambda
    selected_coefficients = cvfit.path.betas[:, best_lambda_index]  # Coefficients for optimal lambda
    return selected_coefficients
end


glmnet_Variable_Selection(X, Y; type="adaptive")

# using LinearAlgebra

# function soft_threshold(rho, alpha)
#     """Soft threshold function used for Lasso regularization"""
#     if rho < -alpha
#         return rho + alpha
#     elseif rho > alpha
#         return rho - alpha
#     else
#         return 0.0
#     end
# end

# function lasso_coordinate_descent(X, y, weights, lambda, tol=1e-4, max_iter=100, adaptive::Bool= false)

#     """
#     Coordinate descent for Lasso regression.

#     Parameters:
#     - X: Input features (matrix).
#     - y: Target values (vector).
#     - lambda: Regularization parameter (float).
#     - tol: Tolerance for stopping criterion (float).
#     - max_iter: Maximum number of iterations (int).

#     Returns:
#     - beta: Coefficients of the Lasso regression model.
#     """
#     n, p = size(X)
#     beta = zeros(p)
#     X_transpose = transpose(X)

#     if adaptive == false
#         for iteration in 1:max_iter
#             max_change = 0.0
#             for j in 1:p
#                 # Compute residual with current coefficients, excluding j
#                 r_j = y - X * beta + X[:, j] * beta[j]
#                 # Update beta_j using soft thresholding
#                 beta_j_new = soft_threshold(dot(X_transpose[j, :], r_j) / n, lambda/ n)
#                 # Update the maximum change in coefficients
#                 max_change = max(max_change, abs(beta_j_new - beta[j]))
#                 # Update beta_j
#                 beta[j] = beta_j_new
#             end
#             # Check for convergence
#             if max_change < tol
#                 break
#             end
#         end
#     else
#         for iteration in 1:max_iter
#             max_change = 0.0
#             for j in 1:p
#                 # Compute residual with current coefficients, excluding j
#                 r_j = y - X * beta + X[:, j] * beta[j]
#                 # Update beta_j using soft thresholding
#                 beta_j_new = soft_threshold(dot(X_transpose[j, :], r_j) / n, lambda/(abs(weights[j])*n))

#                 # Update the maximum change in coefficients
#                 max_change = max(max_change, abs(beta_j_new - beta[j]))
#                 # Update beta_j
#                 beta[j] = beta_j_new
#             end
#             # Check for convergence
#             if max_change < tol
#                 break
#             end
#         end
#     end

#     return beta
# end

# function Lasso_Variable_Selection(X, Y, lasso_type="lasso")
#     # Parameters
#     nγ = 100
#     γM = range(0, stop=400, length=nγ)  # Range of gamma values

#     # Initialize containers
#     n_features = size(X, 2)
#     #Ridge parameters
#     bRidge = fill(NaN, n_features, nγ)       # Ridge coefficients
#     bRidgeMSE = fill(NaN, nγ)               # Ridge Mean Squared Error
    
#     #Lasso parameters
#     bLasso = fill(NaN, n_features, nγ)      # Lasso coefficients
#     bLassoMSE = fill(NaN, nγ)               # Lasso Mean Squared Error
#     Lassolambda = fill(NaN, nγ)             # Lasso lambda values

#     #Adaptive parameters
#     AbLasso = fill(NaN, n_features, nγ)     # Adaptive Lasso coefficients
#     AbLassoMSE = fill(NaN, nγ)              # Adaptive Lasso Mean Squared Error
#     ALassolambda = fill(NaN, nγ)            # Adaptive Lasso lambda values

#     # Ridge Regression
#     if lasso_type == "ridge"
#         for i in 1:nγ
#             Rsol = inv((X'*X -  γM[i]*Diagonal(ones(size(X,2)))))*X'*Y
#             bRidge[:, i] .= Rsol 
#             bRidgeMSE[i] = (Y - X * Rsol)' * (Y - X * Rsol)/ size(X,1)
#         end
#     end

#     # Standard Lasso
#     if lasso_type == "lasso"
#         for i in 1:nγ
#             Lsol = lasso_coordinate_descent(X, Y, bRidge[:,findmin(bRidgeMSE)[2]], γM[i], false)
#             Lassolambda[i] = γM[i]
#             bLasso[:, i] .= Lsol
#             bLassoMSE[i] = ((Y - X * Lsol)' *  (Y - X * Lsol))/ size(X,1)
#         end
#     end

#     # Adaptive Lasso
#     if lasso_type == "adaptive"
#         for i in 1:nγ
#             ALsol = lasso_coordinate_descent(X, Y, bRidge[:,findmin(bRidgeMSE)[2]], γM[i], true)
#             ALassolambda[i] = γM[i]
#             AbLasso[:, i] .= ALsol
#             AbLassoMSE[i] = ((Y - X * ALsol)' *  (Y - X * ALsol))/ size(X,1)
#         end
#     end

#     # Return results based on lasso_type
#     if lasso_type == "ridge"
#         return findall(!iszero,bLasso[:,findmin(bRidgeMSE)[2]])
#     elseif lasso_type == "lasso"
#         return findall(!iszero,bLasso[:,findmin(bLassoMSE)[2]])
#     elseif lasso_type == "adaptive"
#         return findall(!iszero,AbLasso[:,findmin(AbLassoMSE)[2]])

#     else
#         error("Invalid lasso_type. Choose from 'ridge', 'lasso', 'adaptive'.")
#     end
# end


# Lasso_Variable_Selection(X, Y, lasso_type="lasso")