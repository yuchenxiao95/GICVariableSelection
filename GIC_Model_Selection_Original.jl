# Remove duplicate methods 
# List all methods
# methods(GIC_Variable_Selection)


# # Retrieve the first method (replace 1 with the correct index if needed)
# m = methods(GIC_Variable_Selection).ms[2]

# # Delete the method
# Base.delete_method(m)

# # Confirm deletion
# methods(GIC_Variable_Selection)


function GIC_Variable_Selection(X, Y, i, Calculate_GIC, Calculate_GIC_short)
    """
    Fast GIC Variable Selection

    Parameters:
    - X: Input features (matrix).
    - y: Target values (vector).
    - i: Initial Input features with non-zero coefficients (matrix)

    Returns:
    - AIC_list: A list of computed AIC values 
    - AIC_coeff: The corresponding variables with non-zero coefficients
    """
    sets = [1:size(X,2);]                        # get the size of the design matrix for the column
    # Repeat the list n times
    n = 1
    repeated_list = repeat(sets , n)
    GIC_list = zeros(length(repeated_list))               # initialize the AIC list with zeros 
    GIC_coeff =  Vector{Vector{Int64}}(undef, length(sets))     # initialize the AIC coeff vector 

    # initialization
    GIC_coef_sets  = i    
    GIC_c, M_inv = Calculate_GIC(Y, X[:,i])    # only calculate the full matrix inverse once 
    current_X = X[:,i]

    for z in repeated_list
        if  z ∈ GIC_coef_sets     # this is equivalent to asking if beta for column z should be zero instead of non-zero
            
            # find the index of z column         
            index = findfirst(x -> x == z, GIC_coef_sets)

            # current design matrix after removing the z column
            GIC_coef_sets_temp = deleteat!(deepcopy(GIC_coef_sets), index)
            X_subsets = X[:,GIC_coef_sets_temp]  
            
            # compute the separate parts for the inverse
            A_hat = M_inv[setdiff(1:end, index), setdiff(1:end, index)] 
            B_hat = M_inv[setdiff(1:end, index), collect(index)]
            C_hat = M_inv[collect(index), setdiff(1:end, index)]'
            D_hat = only(M_inv[collect(index),collect(index)])
            A_inv = A_hat - ((B_hat / D_hat) * C_hat)

            GIC_i = Calculate_GIC_short(Y, X_subsets, A_inv)   # compute the AIC without z column 
                
            # if AIC with z column is lower than the ACI without the z column than column z has coeff zero 
            if (GIC_i < GIC_c )                    
                GIC_list[z] = GIC_c
            else                     # otherwise add the z column to the design matrix 
                GIC_c = GIC_i
                GIC_list[z] = GIC_c
                GIC_coef_sets = GIC_coef_sets_temp
                M_inv = A_inv
                current_X = X_subsets 
            end

        elseif z ∉ GIC_coef_sets      # if beta for column z is already non-zero, we test whether remove it would be the better or same. 
                
            GIC_coef_sets_temp =  prepend!(deepcopy.(GIC_coef_sets), z)
            X_subsets = X[:,GIC_coef_sets_temp]
            
            index = findfirst(x -> x == z,  GIC_coef_sets_temp)

            # comoute the X'X
            Xsquare = X_subsets' * X_subsets
            
            A_hat = Xsquare[setdiff(1:end, index), setdiff(1:end, index)] 
            B_hat = Xsquare[setdiff(1:end, index), collect(index)]
            C_hat = Xsquare[collect(index), setdiff(1:end, index)]'
            D_hat = only(Xsquare[collect(index),collect(index)])

            # Compute intermediate terms
            topleft = M_inv + M_inv* B_hat *inv(D_hat - C_hat*M_inv*B_hat)*C_hat*M_inv
            topright = -M_inv*B_hat*inv(D_hat-C_hat*M_inv*B_hat)
            bottomleft = -inv(D_hat-C_hat*M_inv*B_hat)*C_hat*M_inv
            bottomright = inv(D_hat - C_hat*M_inv*B_hat)
            # Construct the 2x2 block matrix
            A_inv = [
                topleft                      topright;
                bottomleft                   bottomright
            ]

            GIC_i = Calculate_GIC_short(Y, X_subsets, A_inv)

            if (GIC_c < GIC_i)
                GIC_c = GIC_i
                GIC_list[z] = GIC_c
                GIC_coef_sets = GIC_coef_sets_temp
                M_inv = A_inv
                current_X = X_subsets 
            else
                GIC_list[z] = GIC_c
            end
        end
   
        GIC_coeff[z] = deepcopy.(GIC_coef_sets)
    end
    return (GIC_list, GIC_coeff)
end



