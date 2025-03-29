using Pkg, DelimitedFiles, LinearAlgebra, Statistics, Plots, Convex, SCS, Distributions, Combinatorics, ProximalBase, Base
using DataFrames, Lasso, GLMNet, Plots, LinearAlgebra, LaTeXStrings, Random, StatsBase, StatsPlots, DataFrames


function soft_threshold(rho, alpha)
    """Soft threshold function used for Lasso regularization"""
    if rho < -alpha
        return rho + alpha
    elseif rho > alpha
        return rho - alpha
    else
        return 0.0
    end
end

function lasso_coordinate_descent(X, y, weights, lambda, tol=1e-4, max_iter=100, adaptive::Bool= false)

    """
    Coordinate descent for Lasso regression.

    Parameters:
    - X: Input features (matrix).
    - y: Target values (vector).
    - lambda: Regularization parameter (float).
    - tol: Tolerance for stopping criterion (float).
    - max_iter: Maximum number of iterations (int).

    Returns:
    - beta: Coefficients of the Lasso regression model.
    """
    n, p = size(X)
    beta = zeros(p)
    X_transpose = transpose(X)

    if adaptive == false
        for iteration in 1:max_iter
            max_change = 0.0
            for j in 1:p
                # Compute residual with current coefficients, excluding j
                r_j = y - X * beta + X[:, j] * beta[j]
                # Update beta_j using soft thresholding
                beta_j_new = soft_threshold(dot(X_transpose[j, :], r_j) / n, lambda/ n)
                # Update the maximum change in coefficients
                max_change = max(max_change, abs(beta_j_new - beta[j]))
                # Update beta_j
                beta[j] = beta_j_new
            end
            # Check for convergence
            if max_change < tol
                break
            end
        end
    else
        for iteration in 1:max_iter
            max_change = 0.0
            for j in 1:p
                # Compute residual with current coefficients, excluding j
                r_j = y - X * beta + X[:, j] * beta[j]
                # Update beta_j using soft thresholding
                beta_j_new = soft_threshold(dot(X_transpose[j, :], r_j) / n, lambda/(abs(weights[j])*n))

                # Update the maximum change in coefficients
                max_change = max(max_change, abs(beta_j_new - beta[j]))
                # Update beta_j
                beta[j] = beta_j_new
            end
            # Check for convergence
            if max_change < tol
                break
            end
        end
    end

    return beta
end


"""
Calculate AIC
"""
function Calculate_AIC(Y, X)

    (T, K) = (size(X, 1), size(X, 2))
    Inverse = inv(X'*X)
    Beta_hat = Inverse*X'*Y
    SSE = Y'*Y - Y'*X*Beta_hat
    #SSE = Y'*(Matrix(1.0I, T, T) - X*inv(X'*X)*X')*Y
    AIC = T*log(SSE/T) + 2*K

    return (AIC,Inverse)
end

function Calculate_AIC_short(Y, X, Inverse)

    (T, K) = (size(X, 1), size(X, 2))
    Beta_hat =  Inverse*X'*Y
    SSE = Y'*Y - Y'*X*Beta_hat
    #SSE = Y'*(Matrix(1.0I, T, T) - X*inv(X'*X)*X')*Y
    AIC = T*log(SSE/T) + 2*K

    return (AIC)
end

"""
Calculate BIC
"""

function Calculate_BIC(Y, X)

    (T, K) = (size(X, 1), size(X, 2))
    Inverse = inv(X'*X)
    Beta_hat = Inverse*X'*Y
    SSE = Y'*Y - Y'*X*Beta_hat
    #SSE = Y'*(Matrix(1.0I, T, T) - X*inv(X'*X)*X')*Y

    BIC = T*log(SSE/T) + K * log(T)

    return (BIC, Inverse)
end

function Calculate_BIC_short(Y, X, Inverse)

    (T, K) = (size(X, 1), size(X, 2))
    Beta_hat =  Inverse*X'*Y
    SSE = Y'*Y - Y'*X*Beta_hat
    #SSE = Y'*(Matrix(1.0I, T, T) - X*inv(X'*X)*X')*Y

    BIC = T*log(SSE/T) + K * log(T)

    #BIC = T*log(SSE/T) + K * log(2000)

    return (BIC)
end


# function AIC_Variable_Selection(X, Y)

    # Random.seed!(1234)
    # true_beta = zeros(size(X,2))
    # true_beta[[1]] = [beta_1]
    # #true_beta[[2]] = [beta_2]
    # Y = X*true_beta + rand(Normal(), size(X,1))

    #----------------------------------------------------------------------------------------------
    # sets = [1:size(X,2);]  
 
    # AIC_list = zeros(size(X,2)) 
    # AIC_coeff =  Vector{Vector{Int64}}(undef, length(sets)) 
    # AIC_coef_sets = Array(1:size(X,2))   
       
    # i = 1 
    # j = 1 
    # while j <= length(sets) 
       
    #         X_subsets = X[:,1:end .!=i] 
    #         AIC_e, M_inv = Calculate_AIC(Y, X) 
 
    #         A_hat = M_inv[1:end .!=i,1:end .!=i]  
    #         B_hat = M_inv[1:end .!=i,i] 
    #         C_hat = M_inv[i,1:end .!=i]' 
    #         D_hat = M_inv[i,i] 
    #         A_inv = A_hat - ((B_hat / D_hat) .* C_hat) 
    #         AIC_i = Calculate_AIC_short(Y, X_subsets, A_inv) 
            
    #         #AIC_i = Calculate_AIC(Y, X_subsets)

    #         if (AIC_e < AIC_i)
    #             X = X
    #             AIC_b = AIC_e
    #             AIC_list[j] = AIC_b
    #             i +=1
    #         else
    #             X = X_subsets
    #             AIC_b = AIC_i
    #             AIC_list[j] = AIC_b
    #             AIC_coef_sets = deleteat!(AIC_coef_sets, AIC_coef_sets .== j)
    #             i = i 
    #             #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
    #         end
            
    #     AIC_coeff[j] = deepcopy.(AIC_coef_sets)
    #     j +=1
    #     #i += 1
        
#     end
#     return (AIC_list, AIC_coeff)

# end
"""
CAICF Consistent AIC with Fisher Information
"""

function Calculate_CAICF(AIC,Y,X)

    (T, K) = (size(X, 1), size(X, 2))
    CAICF  = AIC + K * log(T) + log(det((X'*X)/ var(Y)))
    return CAICF
end


"""
CAICF Variable Selection
"""
function CAICF_Variable_Selection(X, Y, i)

    """
    Fast AIC Variable Selection

    Parameters:
    - X: Input features (matrix).
    - y: Target values (vector).
    - i: Initial Input features with non-zero coefficients (matrix)

    Returns:
    - AIC_list: A list of computed AIC values 
    - AIC_coeff: The corresponding variables with non-zero coefficients
    """
    sets = [1:size(X,2);]                        # get the size of the design matrix for the column
    CAICF_list = zeros(length(sets))               # initialize the AIC list with zeros 
    CAICF_coeff =  Vector{Vector{Int64}}(undef, length(sets))     # initialize the AIC coeff vector 
    

    CAICF_coef_sets  = setdiff(sets, i)           
    #BIC_coef_sets = Array(1:size(X[:,setdiff(1:end, i)],2))   
    z = 1                             # iterator for each col in the full design matrix 
    j = 1                             # iterator for the AIC_list
    AIC_t, M_inv = Calculate_AIC(Y, X)    # only calculate the full matrix inverse once 
    while j <= length(sets)

            if  z ∈ i     # this is equivalent to asking if beta for column z should be zero instead of non-zero
                i_c = setdiff(i, [z])  
                X_subsets = X[:,setdiff(1:end, i)]           # design matrix without z column
                current_X = X[:,setdiff(1:end, i_c)]         # design matrix with z column 
                #current_X = hcat(X_subsets[:, 1:z-1], X[:, z], X_subsets[:, z:end])
              
                A_hat = M_inv[setdiff(1:end, i), setdiff(1:end, i)] 
                B_hat = M_inv[setdiff(1:end, i), collect(i)]
                C_hat = M_inv[collect(i), setdiff(1:end, i)]
                D_hat = M_inv[collect(i),collect(i)]
                A_inv = A_hat - ((B_hat / D_hat) * C_hat)
                AIC_i = Calculate_AIC_short(Y, X_subsets, A_inv)   # compute the AIC without z column 
                CAICF_i = Calculate_CAICF(AIC_i, Y, X_subsets)

                A_hat2 = M_inv[setdiff(1:end, i_c), setdiff(1:end, i_c)] 
                B_hat2 = M_inv[setdiff(1:end, i_c), collect(i_c)]
                C_hat2 = M_inv[collect(i_c), setdiff(1:end, i_c)]
                D_hat2 = M_inv[collect(i_c),collect(i_c)]
                A_inv2 = A_hat2 - ((B_hat2 / D_hat2) * C_hat2)
                AIC_e = Calculate_AIC_short(Y, current_X, A_inv2)   # compute the AIC with z column
                CAICF_e = Calculate_CAICF(AIC_e, Y, current_X)

                # if AIC with z column is lower than the ACI without the z column than column z has coeff zero 
                if (CAICF_i - 2 <= CAICF_e )                    
                    X_subsets = X_subsets
                    CAICF_b = CAICF_i
                    CAICF_list[j] = CAICF_b
                
                else                     # otherwise add the z column to the design matrix 
                    X_subsets = current_X
                    CAICF_b = CAICF_e
                    CAICF_list[j] = CAICF_b
                    CAICF_coef_sets = push!(CAICF_coef_sets, z)
                    #BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== j)
                    #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
                end
            elseif z ∉ i      # if beta for column z is already non-zero, we test whether remove it would be the better or same. 
                 
                i_c =  prepend!(deepcopy.(i), z)
                X_subsets = X[:,setdiff(1:end, i)]
                current_X = X[:,setdiff(1:end, i_c)]

                A_hat = M_inv[setdiff(1:end, i), setdiff(1:end, i)] 
                B_hat = M_inv[setdiff(1:end, i), collect(i)]
                C_hat = M_inv[collect(i), setdiff(1:end, i)]
                D_hat = M_inv[collect(i),collect(i)]
                A_inv = A_hat - ((B_hat / D_hat) * C_hat)
                AIC_i = Calculate_AIC_short(Y, X_subsets, A_inv)
                CAICF_i = Calculate_CAICF(AIC_i, Y, X_subsets)

                A_hat2 = M_inv[setdiff(1:end, i_c), setdiff(1:end, i_c)] 
                B_hat2 = M_inv[setdiff(1:end, i_c), collect(i_c)]
                C_hat2 = M_inv[collect(i_c), setdiff(1:end, i_c)]
                D_hat2 = M_inv[collect(i_c),collect(i_c)]
                A_inv2 = A_hat2 - ((B_hat2 / D_hat2) * C_hat2)
                AIC_e = Calculate_AIC_short(Y, current_X, A_inv2)
                CAICF_e = Calculate_CAICF(AIC_e, Y, current_X)

                if (CAICF_e - 2 <= CAICF_i)
                    X_subsets = current_X
                    CAICF_b = CAICF_e
                    CAICF_list[j] = CAICF_b
                    CAICF_coef_sets = deleteat!(CAICF_coef_sets, CAICF_coef_sets .== z)
                    #BIC_coef_sets = push!(BIC_coef_sets, z)
                else
                    X_subsets = X_subsets
                    CAICF_b = CAICF_i
                    CAICF_list[j] = CAICF_b
                    #BIC_coef_sets = push!(BIC_coef_sets, z)
                end
            end

            CAICF_coeff[j] = deepcopy.(CAICF_coef_sets)
            z += 1
            j += 1
            #BIC_i, M_inv = Calculate_BIC(Y, X_subsets)
            # if (BIC_e < BIC_i)
            #     X = X
            #     BIC_b = BIC_e
            #     BIC_list[j] = BIC_b
            #     i +=1
            # else
            #     X = X_subsets
            #     BIC_b = BIC_i
            #     BIC_list[j] = BIC_b
            #     BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== j)
            #     #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
            #     i = i
            # end

    end
    return (CAICF_list, CAICF_coeff)

end


"""
SIC Variable Selection
"""
function SIC_Variable_Selection(X, Y, i)

    """
    Fast SIC Variable Selection

    Parameters:
    - X: Input features (matrix).
    - y: Target values (vector).
    - i: Initial Input features with non-zero coefficients (matrix)

    Returns:
    - AIC_list: A list of computed AIC values 
    - AIC_coeff: The corresponding variables with non-zero coefficients
    """
    sets = [1:size(X,2);]                        # get the size of the design matrix for the column
    SIC_list = zeros(length(sets))               # initialize the AIC list with zeros 
    SIC_coeff =  Vector{Vector{Int64}}(undef, length(sets))     # initialize the AIC coeff vector 
    

    SIC_coef_sets  = setdiff(sets, i)           
    #BIC_coef_sets = Array(1:size(X[:,setdiff(1:end, i)],2))   
    z = 1                             # iterator for each col in the full design matrix 
    j = 1                             # iterator for the AIC_list
    SIC_t, M_inv = Calculate_SIC(Y, X)    # only calculate the full matrix inverse once 
    while j <= length(sets)

            if  z ∈ i     # this is equivalent to asking if beta for column z should be zero instead of non-zero
                i_c = setdiff(i, [z])  
                X_subsets = X[:,setdiff(1:end, i)]           # design matrix without z column
                current_X = X[:,setdiff(1:end, i_c)]         # design matrix with z column 
                #current_X = hcat(X_subsets[:, 1:z-1], X[:, z], X_subsets[:, z:end])
              
                A_hat = M_inv[setdiff(1:end, i), setdiff(1:end, i)] 
                B_hat = M_inv[setdiff(1:end, i), collect(i)]
                C_hat = M_inv[collect(i), setdiff(1:end, i)]
                D_hat = M_inv[collect(i),collect(i)]
                A_inv = A_hat - ((B_hat / D_hat) * C_hat)
                SIC_i = Calculate_SIC_short(Y, X_subsets, A_inv)   # compute the AIC without z column 

                A_hat2 = M_inv[setdiff(1:end, i_c), setdiff(1:end, i_c)] 
                B_hat2 = M_inv[setdiff(1:end, i_c), collect(i_c)]
                C_hat2 = M_inv[collect(i_c), setdiff(1:end, i_c)]
                D_hat2 = M_inv[collect(i_c),collect(i_c)]
                A_inv2 = A_hat2 - ((B_hat2 / D_hat2) * C_hat2)
                SIC_e = Calculate_SIC_short(Y, current_X, A_inv2)   # compute the AIC with z column
                 
                # if AIC with z column is lower than the ACI without the z column than column z has coeff zero 
                if (SIC_i - 2 <= SIC_e )                    
                    X_subsets = X_subsets
                    SIC_b = SIC_i
                    SIC_list[j] = SIC_b
                
                else                     # otherwise add the z column to the design matrix 
                    X_subsets = current_X
                    SIC_b = SIC_e
                    SIC_list[j] = SIC_b
                    SIC_coef_sets = push!(SIC_coef_sets, z)
                    #BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== j)
                    #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
                end
            elseif z ∉ i      # if beta for column z is already non-zero, we test whether remove it would be the better or same. 
                 
                i_c =  prepend!(deepcopy.(i), z)
                X_subsets = X[:,setdiff(1:end, i)]
                current_X = X[:,setdiff(1:end, i_c)]

                A_hat = M_inv[setdiff(1:end, i), setdiff(1:end, i)] 
                B_hat = M_inv[setdiff(1:end, i), collect(i)]
                C_hat = M_inv[collect(i), setdiff(1:end, i)]
                D_hat = M_inv[collect(i),collect(i)]
                A_inv = A_hat - ((B_hat / D_hat) * C_hat)
                SIC_i = Calculate_SIC_short(Y, X_subsets, A_inv)

                A_hat2 = M_inv[setdiff(1:end, i_c), setdiff(1:end, i_c)] 
                B_hat2 = M_inv[setdiff(1:end, i_c), collect(i_c)]
                C_hat2 = M_inv[collect(i_c), setdiff(1:end, i_c)]
                D_hat2 = M_inv[collect(i_c),collect(i_c)]
                A_inv2 = A_hat2 - ((B_hat2 / D_hat2) * C_hat2)
                SIC_e = Calculate_SIC_short(Y, current_X, A_inv2)
                
                if (SIC_e - 2 <= SIC_i)
                    X_subsets = current_X
                    SIC_b = SIC_e
                    SIC_list[j] = SIC_b
                    SIC_coef_sets = deleteat!(SIC_coef_sets, SIC_coef_sets .== z)
                    #BIC_coef_sets = push!(BIC_coef_sets, z)
                else
                    X_subsets = X_subsets
                    SIC_b = SIC_i
                    SIC_list[j] = SIC_b
                    #BIC_coef_sets = push!(BIC_coef_sets, z)
                end
            end

            SIC_coeff[j] = deepcopy.(SIC_coef_sets)
            z += 1
            j += 1
            #BIC_i, M_inv = Calculate_BIC(Y, X_subsets)
            # if (BIC_e < BIC_i)
            #     X = X
            #     BIC_b = BIC_e
            #     BIC_list[j] = BIC_b
            #     i +=1
            # else
            #     X = X_subsets
            #     BIC_b = BIC_i
            #     BIC_list[j] = BIC_b
            #     BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== j)
            #     #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
            #     i = i
            # end

    end
    return (SIC_list, SIC_coeff)

end

"""
AIC Variable Selection
"""
function AIC_Variable_Selection(X, Y, i)

    """
    Fast AIC Variable Selection

    Parameters:
    - X: Input features (matrix).
    - y: Target values (vector).
    - i: Initial Input features with non-zero coefficients (matrix)

    Returns:
    - AIC_list: A list of computed AIC values 
    - AIC_coeff: The corresponding variables with non-zero coefficients
    """
    sets = [1:size(X,2);]                        # get the size of the design matrix for the column
    AIC_list = zeros(length(sets))               # initialize the AIC list with zeros 
    AIC_coeff =  Vector{Vector{Int64}}(undef, length(sets))     # initialize the AIC coeff vector 
    

    AIC_coef_sets  = setdiff(sets, i)           
    #BIC_coef_sets = Array(1:size(X[:,setdiff(1:end, i)],2))   
    z = 1                             # iterator for each col in the full design matrix 
    j = 1                             # iterator for the AIC_list
    AIC_t, M_inv = Calculate_AIC(Y, X)    # only calculate the full matrix inverse once 
    while j <= length(sets)

            if  z ∈ i     # this is equivalent to asking if beta for column z should be zero instead of non-zero
                i_c = setdiff(i, [z])  
                X_subsets = X[:,setdiff(1:end, i)]           # design matrix without z column
                current_X = X[:,setdiff(1:end, i_c)]         # design matrix with z column 
                #current_X = hcat(X_subsets[:, 1:z-1], X[:, z], X_subsets[:, z:end])
              
                A_hat = M_inv[setdiff(1:end, i), setdiff(1:end, i)] 
                B_hat = M_inv[setdiff(1:end, i), collect(i)]
                C_hat = M_inv[collect(i), setdiff(1:end, i)]
                D_hat = M_inv[collect(i),collect(i)]
                A_inv = A_hat - ((B_hat / D_hat) * C_hat)
                AIC_i = Calculate_AIC_short(Y, X_subsets, A_inv)   # compute the AIC without z column 

                A_hat2 = M_inv[setdiff(1:end, i_c), setdiff(1:end, i_c)] 
                B_hat2 = M_inv[setdiff(1:end, i_c), collect(i_c)]
                C_hat2 = M_inv[collect(i_c), setdiff(1:end, i_c)]
                D_hat2 = M_inv[collect(i_c),collect(i_c)]
                A_inv2 = A_hat2 - ((B_hat2 / D_hat2) * C_hat2)
                AIC_e = Calculate_AIC_short(Y, current_X, A_inv2)   # compute the AIC with z column
                 
                # if AIC with z column is lower than the ACI without the z column than column z has coeff zero 
                if (AIC_i - 2 <= AIC_e )                    
                    X_subsets = X_subsets
                    AIC_b = AIC_i
                    AIC_list[j] = AIC_b
                
                else                     # otherwise add the z column to the design matrix 
                    X_subsets = current_X
                    AIC_b = AIC_e
                    AIC_list[j] = AIC_b
                    AIC_coef_sets = push!(AIC_coef_sets, z)
                    #BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== j)
                    #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
                end
            elseif z ∉ i      # if beta for column z is already non-zero, we test whether remove it would be the better or same. 
                 
                i_c =  prepend!(deepcopy.(i), z)
                X_subsets = X[:,setdiff(1:end, i)]
                current_X = X[:,setdiff(1:end, i_c)]

                A_hat = M_inv[setdiff(1:end, i), setdiff(1:end, i)] 
                B_hat = M_inv[setdiff(1:end, i), collect(i)]
                C_hat = M_inv[collect(i), setdiff(1:end, i)]
                D_hat = M_inv[collect(i),collect(i)]
                A_inv = A_hat - ((B_hat / D_hat) * C_hat)
                AIC_i = Calculate_AIC_short(Y, X_subsets, A_inv)

                A_hat2 = M_inv[setdiff(1:end, i_c), setdiff(1:end, i_c)] 
                B_hat2 = M_inv[setdiff(1:end, i_c), collect(i_c)]
                C_hat2 = M_inv[collect(i_c), setdiff(1:end, i_c)]
                D_hat2 = M_inv[collect(i_c),collect(i_c)]
                A_inv2 = A_hat2 - ((B_hat2 / D_hat2) * C_hat2)
                AIC_e = Calculate_AIC_short(Y, current_X, A_inv2)
                
                if (AIC_e - 2 <= AIC_i)
                    X_subsets = current_X
                    AIC_b = AIC_e
                    AIC_list[j] = AIC_b
                    AIC_coef_sets = deleteat!(AIC_coef_sets, AIC_coef_sets .== z)
                    #BIC_coef_sets = push!(BIC_coef_sets, z)
                else
                    X_subsets = X_subsets
                    AIC_b = AIC_i
                    AIC_list[j] = AIC_b
                    #BIC_coef_sets = push!(BIC_coef_sets, z)
                end
            end

            AIC_coeff[j] = deepcopy.(AIC_coef_sets)
            z += 1
            j += 1
            #BIC_i, M_inv = Calculate_BIC(Y, X_subsets)
            # if (BIC_e < BIC_i)
            #     X = X
            #     BIC_b = BIC_e
            #     BIC_list[j] = BIC_b
            #     i +=1
            # else
            #     X = X_subsets
            #     BIC_b = BIC_i
            #     BIC_list[j] = BIC_b
            #     BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== j)
            #     #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
            #     i = i
            # end

    end
    return (AIC_list, AIC_coeff)

end


"""
BIC Variable Selection
"""
function BIC_Variable_Selection(X, Y, i)


    # Random.seed!(1234)
    # true_beta = zeros(size(X,2))
    # true_beta[[1]] = [beta_1]
    # #true_beta[[2]] = [beta_2]
    # Y = X*true_beta + rand(Normal(), size(X,1))

    #----------------------------------------------------------------------------------------------
   
    sets = [1:size(X,2);] 
    BIC_list = zeros(length(sets))
    BIC_coeff =  Vector{Vector{Int64}}(undef, length(sets))
    
    #i = [2,3,4,5,16]                  # the sets of columns not in the design matrix
    BIC_coef_sets  = setdiff(sets, i)
    #BIC_coef_sets  = i
    z = 1                            # iterator for each col in the full design matrix 
    j = 1                             # iterator for the BIC_list
    BIC_t, M_inv = Calculate_BIC(Y, X)
    while j <= length(sets)
      
            # X_subsets = X[:,1:end .!=i]
            # BIC_e, M_inv = Calculate_BIC(Y, X)

            # A_hat = M_inv[1:end .!=i, 1:end .!=i] 
            # B_hat = M_inv[1:end .!=i, i]
            # C_hat = M_inv[i, 1:end .!=i]'
            # D_hat = M_inv[i,i]
            # A_inv = A_hat - ((B_hat / D_hat) .* C_hat)
            # BIC_i = Calculate_BIC_short(Y, X_subsets, A_inv)
            
            if  z ∈ i           # this is equivalent to asking beta for column z is zero, 
                                #then we test whether add column z would be better

                #i_c =  prepend!(deepcopy.(i), z)
                i_c = setdiff(i, [z])
                X_subsets = X[:,setdiff(1:end, i)]
                current_X = X[:,setdiff(1:end, i_c)]
 
                ##current_X = hcat(X_subsets[:, 1:z-1], X[:, z], X_subsets[:, z:end])
              
                A_hat = M_inv[setdiff(1:end, i), setdiff(1:end, i)] 
                B_hat = M_inv[setdiff(1:end, i), collect(i)]
                C_hat = M_inv[collect(i), setdiff(1:end, i)]
                D_hat = M_inv[collect(i),collect(i)]
                A_inv = A_hat - ((B_hat / D_hat) * C_hat)
                BIC_i = Calculate_BIC_short(Y, X_subsets, A_inv)

                A_hat2 = M_inv[setdiff(1:end, i_c), setdiff(1:end, i_c)] 
                B_hat2 = M_inv[setdiff(1:end, i_c), collect(i_c)]
                C_hat2 = M_inv[collect(i_c), setdiff(1:end, i_c)]
                D_hat2 = M_inv[collect(i_c),collect(i_c)]
                A_inv2 = A_hat2 - ((B_hat2 / D_hat2) * C_hat2)
                BIC_e = Calculate_BIC_short(Y, current_X, A_inv2)
                
                if (BIC_i <= BIC_e)
                    X_subsets = X_subsets
                    BIC_b = BIC_i
                    BIC_list[j] = BIC_b
                
                else
                    X_subsets = current_X
                    BIC_b = BIC_e
                    BIC_list[j] = BIC_b
                    BIC_coef_sets = push!(BIC_coef_sets, z)
                    #BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== j)
                    #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
                    #i = i
                end
            elseif z ∉ i      # if beta for column z is already one, we test whether remove it would be the better or same. 
                 
                i_c =  prepend!(deepcopy.(i), z)
                X_subsets = X[:,setdiff(1:end, i)]
                current_X = X[:,setdiff(1:end, i_c)]

                A_hat = M_inv[setdiff(1:end, i), setdiff(1:end, i)] 
                B_hat = M_inv[setdiff(1:end, i), collect(i)]
                C_hat = M_inv[collect(i), setdiff(1:end, i)]
                D_hat = M_inv[collect(i),collect(i)]
                A_inv = A_hat - ((B_hat / D_hat) * C_hat)
                BIC_i = Calculate_BIC_short(Y, X_subsets, A_inv)

                A_hat2 = M_inv[setdiff(1:end, i_c), setdiff(1:end, i_c)] 
                B_hat2 = M_inv[setdiff(1:end, i_c), collect(i_c)]
                C_hat2 = M_inv[collect(i_c), setdiff(1:end, i_c)]
                D_hat2 = M_inv[collect(i_c),collect(i_c)]
                A_inv2 = A_hat2 - ((B_hat2 / D_hat2) * C_hat2)
                BIC_e = Calculate_BIC_short(Y, current_X, A_inv2)
                
                if (BIC_e <= BIC_i)
                    X_subsets = current_X
                    BIC_b = BIC_e
                    BIC_list[j] = BIC_b
                    BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== z)
                    #BIC_coef_sets = push!(BIC_coef_sets, z)
                else
                    X_subsets = X_subsets
                    BIC_b = BIC_i
                    BIC_list[j] = BIC_b
                    #BIC_coef_sets = push!(BIC_coef_sets, z)
                end
            end

            BIC_coeff[j] = deepcopy.(BIC_coef_sets)
            z += 1
            j += 1
            #BIC_i, M_inv = Calculate_BIC(Y, X_subsets)
            # if (BIC_e < BIC_i)
            #     X = X
            #     BIC_b = BIC_e
            #     BIC_list[j] = BIC_b
            #     i +=1
            # else
            #     X = X_subsets
            #     BIC_b = BIC_i
            #     BIC_list[j] = BIC_b
            #     BIC_coef_sets = deleteat!(BIC_coef_sets, BIC_coef_sets .== j)
            #     #deleteat!(coef_sets, findall(x ->x == i, coef_sets))
            #     i = i
            # end

    end
    return (BIC_list, BIC_coeff)

end

function Lasso_Variable_Selection(X, Y)

    # Random.seed!(1234)
    # true_beta = zeros(size(X,2))
    # true_beta[[1]] = [beta_1]
    # #true_beta[[2]] = [beta_2]
    # Y = X*true_beta + rand(Normal(), size(X,1))

    nγ = 100
    γM = range(0; stop = 400, length = nγ)             #different γ values
    # lasso parameters
    bLasso = fill(NaN, size(X, 2), nγ)       #results for γM[i] are in bLasso[:,i]
    bLassoMSE = fill(NaN, nγ)    
    Lassolambda = fill(NaN, nγ)
    # adaptive lasso parameter
    AbLasso = fill(NaN, size(X, 2), nγ)       #results for γM[i] are in bLasso[:,i]
    AbLassoMSE = fill(NaN, nγ)    
    ALassolambda = fill(NaN, nγ)

    # ridge parameters
    bRidge = fill(NaN, size(X, 2), nγ)
    bRidgeMSE = fill(NaN, nγ)  
    
    # to compute the ridge estimates and use them as weights for adaptive lasso
    for i in 1:nγ
        #ridge
        Rsol = inv((X'*X -  γM[i]*Diagonal(ones(size(X,2)))))*X'*Y
        bRidge[:, i] .= Rsol 
        bRidgeMSE[i] = (Y - X * Rsol)' * (Y - X * Rsol)/ size(X,1)
    end

    # Lasso 
    for i in 1:nγ
        # lasso
        Lsol = lasso_coordinate_descent(X, Y, bRidge[:,findmin(bRidgeMSE)[2]], γM[i], false)
        Lassolambda[i] = γM[i]
        bLasso[:, i] .= Lsol
        bLassoMSE[i] = ((Y - X * Lsol)' *  (Y - X * Lsol))/ size(X,1)
    end
    # adaptive lasso
    for i in 1:nγ
        # lasso
        ALsol = lasso_coordinate_descent(X, Y, bRidge[:,findmin(bRidgeMSE)[2]], γM[i], true)
        ALassolambda[i] = γM[i]
        AbLasso[:, i] .= ALsol
        AbLassoMSE[i] = ((Y - X * ALsol)' *  (Y - X * ALsol))/ size(X,1)
    end
    #lasso_coordinate_descent(X, y, weights,  lambda, tol=1e-4, max_iter=100, adaptive = false)

    #bLasso[:,findmin(bLassoMSE)[2]]

    return  (findall(!iszero,bLasso[:,findmin(bLassoMSE)[2]]), findall(!iszero,AbLasso[:,findmin(AbLassoMSE)[2]]))

    #Lassolambda[findmin(bLassoMSE)[2]]
end



######################


#################################################################
### The main function that call individual IC for model selection
#################################################################

function Variable_Selection(X,Y,random_initial_columns)

    Best_AIC_List = Vector{Vector{Int64}}(undef, 10)     
    Best_BIC_List = Vector{Vector{Int64}}(undef, 10)     
    Best_SIC_List = Vector{Vector{Int64}}(undef, 10)     
    Best_CAICF_List = Vector{Vector{Int64}}(undef, 10) 

    sets = [1:size(X,2);] 

    AIC_list, AIC_coeff = AIC_Variable_Selection(X, Y, random_initial_columns)  #random_index_zero_columns
    BIC_list, BIC_coeff = BIC_Variable_Selection(X, Y, random_initial_columns) 
    SIC_list, SIC_coeff = SIC_Variable_Selection(X, Y, random_initial_columns) 
    CAICF_list, CAICF_coeff = CAICF_Variable_Selection(X, Y, random_initial_columns) 
  """
  i = 1
  while i <= 10 
      if i == 1 
        else
            AIC_list, AIC_coeff = AIC_Variable_Selection(X, Y, setdiff(sets, random_initial_columns))  #random_index_zero_columns
            BIC_list, BIC_coeff = BIC_Variable_Selection(X, Y, setdiff(sets, random_initial_columns)) 
            SIC_list, SIC_coeff = SIC_Variable_Selection(X, Y, setdiff(sets, random_initial_columns)) 
            CAICF_list, CAICF_coeff = CAICF_Variable_Selection(X, Y, setdiff(sets, random_initial_columns)) 
        i = i+1
        end
    end
    """
    Best_AIC_List = AIC_coeff[end]
    Best_BIC_List = BIC_coeff[end]
    Best_SIC_List = SIC_coeff[end]
    Best_CAICF_List = CAICF_coeff[end]

    lasso, adaptive_lasso = Lasso_Variable_Selection(X, Y)   


 return(Best_AIC_List, Best_BIC_List, Best_SIC_List, Best_CAICF_List, lasso, adaptive_lasso)

end
########################################################################################################
### Function that output overselection, underselection for different sample size and model dimension
#########################################################################################################
function Overselection(VS, true_columns)

    overselect_parameters_BIC = length(setdiff(VS[2], true_columns))
    underselect_parameters_BIC = length(setdiff(true_columns, VS[2]))
    selected_true_parameters_BIC = length(intersect(VS[2], true_columns))

    overselect_parameters_AIC = length(setdiff(VS[1], true_columns))
    underselect_parameters_AIC = length(setdiff(true_columns, VS[1]))
    selected_true_parameters_AIC = length(intersect(VS[1], true_columns))
    
    overselect_parameters_SIC = length(setdiff(VS[3], true_columns))
    underselect_parameters_SIC = length(setdiff(true_columns, VS[3]))
    selected_true_parameters_SIC = length(intersect(VS[3], true_columns))
    
    overselect_parameters_CAICF = length(setdiff(VS[4], true_columns))
    underselect_parameters_CAICF = length(setdiff(true_columns, VS[4]))
    selected_true_parameters_CAICF = length(intersect(VS[4], true_columns))
    
    overselect_parameters_Lasso= length(setdiff(VS[5], true_columns))
    underselect_parameters_Lasso = length(setdiff(true_columns, VS[5]))
    selected_true_parameters_Lasso = length(intersect(VS[5], true_columns))
    
    overselect_parameters_adaptive_lasso= length(setdiff(VS[6], true_columns))
    underselect_parameters_adaptive_lasso = length(setdiff(true_columns, VS[6]))
    selected_true_parameters_adaptive_lasso = length(intersect(VS[6], true_columns))

    return( overselect_parameters_BIC,underselect_parameters_BIC, selected_true_parameters_BIC,
    overselect_parameters_AIC, underselect_parameters_AIC, selected_true_parameters_AIC,
    overselect_parameters_SIC, underselect_parameters_SIC, selected_true_parameters_SIC,
    overselect_parameters_CAICF, underselect_parameters_CAICF, selected_true_parameters_CAICF,
    overselect_parameters_Lasso, underselect_parameters_Lasso, selected_true_parameters_Lasso,
    overselect_parameters_adaptive_lasso, underselect_parameters_adaptive_lasso, selected_true_parameters_adaptive_lasso)

end

##############################################################################
### Main data generation to create design matrix NxP with k true parameters
##############################################################################
function Generation(N, P, k, true_columns)

mu = zeros(P)              # Set the mean
cov = Diagonal(ones(P))    # Set the covariance
X = rand(MvNormal(mu, cov), N)'     # Generate the design matrix 

coeff_true = append!(rand(Uniform(3,6), length(true_columns)))     #set the true coefficients
#append!(ones(length(true_columns)-2), 0.3, 0.1, 0.1) 

true_beta = zeros(size(X,2))

for i in 1:length(true_beta)  
    if (i in true_columns)   
        true_beta[i] = coeff_true[indexin(i, true_columns)][1]  
    else   
        true_beta[i] = 0 
    end  
    i +=1  
end 
    
#findall(!iszero, true_beta)    
# Generate the True Y 
Y = X*true_beta + rand(Normal(), size(X,1))

return(X,Y)
end

Random.seed!(9)         # Set the random seed
N = 3000          
P = 100            
k = 15
random_index_true_columns = sort(sample(1:P, k, replace = false))     # randomly select the "true" columns 
X,Y = Generation(N,P,k, random_index_true_columns)

# Random subset to create datasets with small sample size that from the same distribution. 
s_index = sample(1:N, 200)
X_small = X[s_index,:]
m_index = sample(1:N, 600)
X_moderate = X[m_index,:]
l_index = sample(1:N, 1000)
X_large = X[l_index,:]

Y_small = Y[s_index]
Y_moderate = Y[m_index]
Y_large = Y[l_index]

# Model selection on various sample size 
random_initial_columns = sample(1:size(X,2), 10, replace = false)

VS_small = Variable_Selection(X_small,Y_small,random_initial_columns)
VS_moderate = Variable_Selection(X_moderate,Y_moderate,random_initial_columns)
VS_large = Variable_Selection(X_large,Y_large,random_initial_columns)
VS_verylarge = Variable_Selection(X,Y,random_initial_columns)


Overselection(VS_small, random_index_true_columns)
Overselection(VS_moderate, random_index_true_columns)
Overselection(VS_large, random_index_true_columns)
Overselection(VS_verylarge, random_index_true_columns)


####################################################
### Summary and Construct the Table
#####################################################
print(random_index_true_columns)
print(Best_BIC_List[10])
print(Best_AIC_List[10])
print(Best_SIC_List[10])
print(Best_CAICF_List[10])
print(lasso)
print(adaptive_lasso)








##########################################################################
### Testing 
#########################################################################
# for multiple columns removal 
i = [1,3,4,5,8,16]


A = X[:, setdiff(1:end, i)]' * X[:, setdiff(1:end, i)]
B = X[:,setdiff(1:end, i)]' * X[:, collect(i)]
C = X[:, collect(i)]' * X[:,setdiff(1:end, i)]
D = Float64.(X[:, collect(i)]' * X[:, collect(i)])

H_hat = hcat(A[:, 1:i-1], B, A[:, i:end])
M_hat = vcat(H_hat[1:i-1,:], hcat(C[:, 1:i-1], D, C[:, 1:i:end]), H_hat[i:end,:])

M_inv = inv(X'*X)

A_hat = M_inv[setdiff(1:end, i),setdiff(1:end, i)] 
B_hat = M_inv[setdiff(1:end, i), collect(i)]
C_hat = M_inv[collect(i), setdiff(1:end, i)]
D_hat = M_inv[collect(i),collect(i)]

# A_hat = M_inv[1:end .!=i,1:end .!=i] 
# B_hat = M_inv[1:end .!=i,i]
# C_hat = M_inv[i,1:end .!=i]'
# D_hat = M_inv[i,i]

A_inv = A_hat - ((B_hat / D_hat) * C_hat)

M = X'*X

A = M[setdiff(1:end, i), setdiff(1:end, i)]
inv(A)

######################################################################################

# for single column removal 
i =1
A = X[:,1:end .!=i]' * X[:, 1:end .!=i]
B = X[:,1:end .!=i]' * X[:, i]
C = X[:, i]' * X[:,1:end .!=i]
D = Float64.(X[:, i]' * X[:, i])

H_hat = hcat(A[:, 1:i-1], B, A[:, i:end])
M_hat = vcat(H_hat[1:i-1,:], hcat(C[:, 1:i-1], D, C[:, 1:i:end]), H_hat[i:end,:])

M_inv = inv(X'*X)

# A_hat = M_inv[1:end-1, 1:end-1] 
# B_hat = M_inv[1:end-1,end]
# C_hat = M_inv[end,1:end-1]'
# D_hat = M_inv[end,end]

A_hat = M_inv[1:end .!=i,1:end .!=i] 
B_hat = M_inv[1:end .!=i,i]
C_hat = M_inv[i,1:end .!=i]'
D_hat = M_inv[i,i]

A_inv = A_hat - ((B_hat / D_hat) .* C_hat)

M = X'*X
A = M[1:end .!=i,1:end .!=i] 
inv(A)



