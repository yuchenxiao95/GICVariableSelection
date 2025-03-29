using Random
using LinearAlgebra

function GIC_Variable_Selection(
    X::AbstractMatrix, 
    #Y::AbstractVector, 
    Y::AbstractMatrix, 
    Init_Columns::AbstractVector{Int}, 
    Calculate_GIC, 
    Calculate_GIC_short;
    Nsim::Int64 = 1
)

    # Input validation
    m, n = size(X)
    # if size(Y)[1] != m
    #     error("The number of rows in X must match the length of Y.")
    # end
    # if any(Init_Columns .> n) || any(Init_Columns .< 1)
    #     error("Initial feature indices must be within 1 to size(X, 2).")
    # end

    sets = collect(1:size(X,2))                  # get the size of the design matrix for the column
    # shuffle!(repeated_list)
    repeated_list = repeat(sets, Nsim)

    GIC_coef_sets = Init_Columns
    GIC_c, M_inv = Calculate_GIC(Y, X[:, GIC_coef_sets])
    current_X = X[:, GIC_coef_sets]

    # Prepare outputs
    GIC_list = zeros(length(repeated_list))
    GIC_coeff  = [Int[] for _ in 1:length(repeated_list)]
    w = 1

    # Parallelized loop
    # @sync for z in repeated_list
    #     @async begin

    #         local GIC_i, GIC_coef_sets_temp, A_inv
    for z in repeated_list
        if z in GIC_coef_sets # Test removing a feature
            index = findfirst(==(z), GIC_coef_sets)
            GIC_coef_sets_temp = deleteat!(copy(GIC_coef_sets), index)
            X_subsets = X[:, GIC_coef_sets_temp]

            # Update inverse using block matrix manipulations
            A_hat = M_inv[setdiff(1:end, index), setdiff(1:end, index)]
            B_hat = M_inv[setdiff(1:end, index), index]
            C_hat = M_inv[index, setdiff(1:end, index)]'
            D_hat = only(M_inv[index, index])
            A_inv = A_hat - ((B_hat / D_hat) * C_hat)

            GIC_i = Calculate_GIC_short(Y, X_subsets, A_inv)

            # Debug: Print types and values
            # println("GIC_i type: ", typeof(GIC_i), " value: ", GIC_i)
            # println("GIC_c type: ", typeof(GIC_c), " value: ", GIC_c)

            if (tr(GIC_i) < tr(GIC_c))      
                GIC_list[w] = tr(GIC_c)
            else                     
                GIC_c = GIC_i
                GIC_list[w] = tr(GIC_c)
                GIC_coef_sets = GIC_coef_sets_temp
                M_inv = A_inv
                current_X = X_subsets 
            end
            # Store coefficients
            GIC_coeff[w] = copy(GIC_coef_sets)
            
        elseif z ∉ GIC_coef_sets      # if beta for column z is already non-zero, we test whether remove it would be the better or same. 
            GIC_coef_sets_temp = prepend!(copy(GIC_coef_sets), z)
            X_subsets = X[:, GIC_coef_sets_temp]

            index = findfirst(==(z), GIC_coef_sets_temp)
            Xsquare = X_subsets' * X_subsets

            A_hat = Xsquare[setdiff(1:end, index), setdiff(1:end, index)]
            B_hat = Xsquare[setdiff(1:end, index), index]
            C_hat = Xsquare[index, setdiff(1:end, index)]'
            D_hat = only(Xsquare[index, index])

            topleft = M_inv + M_inv * B_hat * inv(D_hat - C_hat * M_inv * B_hat) * C_hat * M_inv
            topright = -M_inv * B_hat * inv(D_hat - C_hat * M_inv * B_hat)
            bottomleft = -inv(D_hat - C_hat * M_inv * B_hat) * C_hat * M_inv
            bottomright = inv(D_hat - C_hat * M_inv * B_hat)
            A_inv = [
                topleft           topright;
                bottomleft        bottomright
            ]

            GIC_i = Calculate_GIC_short(Y, X_subsets, A_inv)

            # Debug: Print types and values
            # println("GIC_i type: ", typeof(GIC_i), " value: ", GIC_i)
            # println("GIC_c type: ", typeof(GIC_c), " value: ", GIC_c)

            if (tr(GIC_c) < tr(GIC_i))
                GIC_c = GIC_i
                GIC_list[w] = tr(GIC_c)
                GIC_coef_sets = GIC_coef_sets_temp
                M_inv = A_inv
                current_X = X_subsets 
            else
                GIC_list[w] = tr(GIC_c)
            end

            # Store coefficients
            GIC_coeff[w] = copy(GIC_coef_sets)
        end
        w += 1
    end

    return (GIC_list, GIC_coeff)
end