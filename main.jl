using Pkg, DelimitedFiles, LinearAlgebra, Statistics, Plots, Convex, SCS, Distributions, Combinatorics
using ProximalBase, Base, GR, IJulia, SharedArrays, LogExpFunctions, Lasso, GLMNet, Plots, LaTeXStrings
using Random, StatsBase, StatsPlots, DataFrames, DataFramesMeta, Distributed, Metal, BenchmarkTools, Combinatorics
using PrettyTables, CSV
# # List of packages to install
# packages = [
#     "DelimitedFiles", "LinearAlgebra", "Statistics", "Plots", "Convex", "SCS", "Distributions",
#     "Combinatorics", "ProximalBase", "Lasso", "GLMNet", "LaTeXStrings", "Random",
#     "StatsBase", "StatsPlots", "DataFrames", "DataFramesMeta"
# ]

# # Install each package
# for pkg in packages
#     Pkg.add(pkg)
# end

include("GIC_Calculation.jl")
include("Comparison_Function.jl")
include("Evaluation_Function.jl")
include("Generation.jl")
#include("GLMNet_Model_Selection.jl")
include("GIC_Model_Selection.jl")
include("GIC_Model_Selection_Boltzmann.jl")
include("Beta_estimate.jl")
include("LP_to_Y.jl")
include("Y_to_lp.jl")


aic, inverse = Calculate_AIC(Y, X)
Calculate_AIC_short(Y, X, inverse)
#################################################################################
###################################################################################
###################################################################################
init_cols = sort(sample(1:P, Int64(floor(P/20)), replace=false))
true_beta

init_cols  = collect(1:P)
#Y_to_lp(Y, "Poisson")
@time begin 
    AIC_tmp = GIC_Variable_Selection(X, Y, init_cols, Calculate_SIC, Calculate_SIC_short, Nsim =5)
end
###################################################################################
AIC_tmpp = DataFrame(A = AIC_tmp[1], 
                    B = AIC_tmp[2])
#Plots.plot(AIC_tmp[1])
random_index_true_columns = unique(reduce(vcat, multi_beta_true_columns)) 
#Calculate_BIC(Y_to_lp(Y, "Bernoulli"), X[:,true_columns])
print(setdiff(AIC_tmp[2][end],random_index_true_columns))
print(setdiff(random_index_true_columns, AIC_tmp[2][end]))


sic, inverse = Calculate_SIC(Y, X[:,AIC_tmp[2][end]])


Beta_est = inverse * X[:,AIC_tmp[2][end]]' * Y


Beta_estimate(Y, X[:,AIC_tmp[2][end]], inverse)


tmp_list = AIC_tmp[2][end]
Calculate_BIC(Y, X[:,random_index_true_columns])
Calculate_BIC(Y, X[:,AIC_tmp[2][end]])
Calculate_BIC(Y, X[:,filter!(x -> x != 34, tmp_list)])
#####################################################################################################


# Convert X to a DataFrame for easier manipulation
df = DataFrame(X, :auto)
df.Y = Y

# Initialize an empty DataFrame to store results
results_df = DataFrame(Combination = String[], BIC = Float64[], BICT = Float64[])

# Generate combinations and calculate BIC and AIC in one loop
for k in 1:size(X, 2)  # Combinations of size 1 to the number of columns in X
    for combo in combinations(1:size(X, 2), k)  # Generate combinations of size k
        # Subset of X corresponding to the current combination
        X_subset = X[:, combo]
        
        # Calculate BIC using your custom function
        BIC_value = Calculate_BIC(Y, X_subset)
        
        # Create the formula dynamically using column names
        combo_names = names(df)[combo]  # Get the column names for the current combination
        formula_str = "Y ~ " * join(combo_names, " + ")  # Create the formula string
        formula = eval(Meta.parse("@formula($formula_str)"))  # Convert the formula string to a FormulaTerm
        
        # Fit the model
        model = lm(formula, df)
        
        # Calculate AIC using StatsBase
        AIC_value = aic(model)
        
        # Convert the combination to a string
        combo_str = join(combo, ",")
        
        # Add the combination, BIC, and AIC to the DataFrame
        push!(results_df, (combo_str, BIC_value[1], AIC_value))
    end
end

# Display the DataFrame with 8 decimal points using PrettyTables
pretty_table(results_df, formatters = (ft_printf("%.8f", [2, 3])))  # Format BIC and AIC columns to 8 decimal points


##############################################################################################
##############################################################################################
#Y_to_lp(Y, "Normal")
@time begin
    AICB_tmp = GIC_Variable_Selection_Boltzmann(X, Y, init_cols, Calculate_BIC, Calculate_BIC_short, Nsim=10, T = 0.15)
end

AICB_tmpp = DataFrame(A = AICB_tmp[1], 
                    B = AICB_tmp[2])
#AICB_tmpp_unique = unique(AICB_tmpp, :B)
Plots.plot(AICB_tmp[1])
print(setdiff(AICB_tmpp.B[],random_index_true_columns))
print(setdiff(random_index_true_columns,AICB_tmpp.B[160]))




# Normalize probabilities
energy_values = [exp(-v/0.2) for v in values(AIC_tmpp_unique.A/1000)]
# Normalize probabilities
prob_distribution = reverse(energy_values ./ sum(energy_values))

# Create x indices (assumes evenly spaced points)
x_indices = 1:length(energy_values)

# Plot the density values
Plots.plot(x_indices, prob_distribution, label="Density", xlabel="Index", ylabel="Density", lw=2)

p = StatsPlots.histogram(AIC_tmp[1])

print(AIC_tmp[2][500])
energy_values = [v for v in AIC_tmpp_unique.A]
prob_distribution = energy_values ./ sum(energy_values)


p = Plots.plot(AIC_tmp[3])
print(AIC_tmp[3])
sum(AIC_tmp[3] .> 0.0001)


@time begin
AIC_tmp = GIC_Variable_Selection_Parallel(X, Y, collect(1:P), Calculate_AIC, Calculate_AIC_short)
end

@time begin
    BIC_tmp = GIC_Variable_Selection_Parallel(X, Y, collect(1:P), Calculate_BIC, Calculate_BIC_short)
end

@time begin
    SIC_tmp = GIC_Variable_Selection_Parallel(X, Y, collect(1:10), Calculate_SIC, Calculate_SIC_short)
end

@time begin
    CAIC_tmp = GIC_Variable_Selection_Parallel(X, Y, collect(1:P), Calculate_CAIC, Calculate_CAIC_short)
    end

@time begin
CAICF_tmp = GIC_Variable_Selection_Parallel(X, Y, collect(1:P), Calculate_CAICF, Calculate_CAICF_short)
end

#GIC_list, GIC_coeff, GIC_beta_estimate, GIC_beta_variance_estimate
print(random_index_true_columns)
# AIC Results
print(setdiff(AIC_tmp[2][end],random_index_true_columns))
print(setdiff(random_index_true_columns, AIC_tmp[2][end]))

# BIC Results 
print(setdiff(BIC_tmp[2][end],random_index_true_columns))
print(setdiff(random_index_true_columns, BIC_tmp[2][end]))

# SIC Results 
print(setdiff(SIC_tmp[2][end],random_index_true_columns))
print(setdiff(random_index_true_columns, SIC_tmp[2][end]))

# CAIC Results 
print(setdiff(CAIC_tmp[2][end],random_index_true_columns))
print(setdiff(random_index_true_columns, CAIC_tmp[2][end]))

# CAICF Results 
print(setdiff(CAICF_tmp[2][end],random_index_true_columns))
print(setdiff(random_index_true_columns, CAICF_tmp[2][end]))


################################################################################
#Lasso
@time begin
cvfit = glmnetcv(X, Y, alpha = 1.0)  # Lasso Cross-validation
#cvfit = glmnetcv(X, Y, Poisson()) 
best_lambda_index = argmin(cvfit.meanloss)  # Find index of optimal lambda
selected_coefficients = cvfit.path.betas[:, best_lambda_index]  # Coefficients for optimal lambda    
lasso_index = findall(!iszero,selected_coefficients)
end

#adaptive Lasso
@time begin
# Perform Ridge regression for adaptive Lasso weights
ridge_fit = glmnet(X, Y, alpha=0.0)  # Ridge regression
ridge_cvfit = glmnetcv(X, Y, alpha=0.0)  # Cross-validation for Ridge
# Extract Ridge coefficients (exclude intercept)
best_lambda_index = argmin(ridge_cvfit.lambda)  # Find index of optimal lambda
best_ridge_coef = ridge_fit.betas[:, best_lambda_index]  # Coefficients for optimal lambda

# Perform adaptive Lasso with penalty factors based on Ridge coefficients
penalty_factor = 1.0 ./ abs.(best_ridge_coef .+ 1e-5)  # Compute penalty factors
cvfit = glmnetcv(X, Y, alpha = 1.0, penalty_factor=penalty_factor)  # Cross-validation for adaptive Lasso

best_lambda_index = argmin(cvfit.meanloss)  # Find index of optimal lambda
selected_coefficients = cvfit.path.betas[:, best_lambda_index]  # Coefficients for optimal lambda
adaptivelasso_index = findall(!iszero,selected_coefficients[:,2])
end


# Lasso Results
print(setdiff(lasso_index, random_index_true_columns))
print(setdiff(random_index_true_columns, lasso_index))

# Adaptive Results
print(setdiff(adaptivelasso_index, random_index_true_columns))
print(setdiff(random_index_true_columns, adaptivelasso_index))
################################################################################################

#Lasso for multivariate
@time begin
    
    cv_model = glmnetcv(X, Y, MvNormal(),alpha=1) 
    # Best lambda (minimizing MSE)
    best_lambda = cv_model.lambda[argmin(cv_model.meanloss)]
    best_lambda_index = argmin(cv_model.meanloss)
    selected_coefficients = cv_model.path.betas[:,:, best_lambda_index]  # Coefficients for optimal lambda
    # Refit model with best lambda
    final_model = glmnet(X, Y, MvNormal(), alpha=1, lambda=[best_lambda])
    lasso_index = findall(!iszero,selected_coefficients[:,1]) 
end

#Adaptive Lasso for Multivariate
@time begin
    # Perform Ridge regression for adaptive Lasso weights
    ridge_fit = glmnet(X, Y, MvNormal(), alpha=0.0)  # Ridge regression
    ridge_cvfit = glmnetcv(X, Y, MvNormal(), alpha=0.0)  # Cross-validation for Ridge
    # Extract Ridge coefficients (exclude intercept)
    best_lambda_index = argmin(ridge_cvfit.lambda)  # Find index of optimal lambda
    best_ridge_coef = ridge_fit.betas[:,:, best_lambda_index]  # Coefficients for optimal lambda
    
    # Perform adaptive Lasso with penalty factors based on Ridge coefficients
    feature_importance = vec(sqrt.(sum(best_ridge_coef.^2, dims=2)))
    penalty_factor =  1 ./ (abs.( feature_importance ) .+ 1e-6)  # Compute penalty factors
    cvfit = glmnetcv(X, Y, MvNormal(), alpha = 1.0, penalty_factor=penalty_factor)  # Cross-validation for adaptive Lasso
    
    best_lambda_index = argmin(cvfit.meanloss)  # Find index of optimal lambda
    selected_coefficients = cvfit.path.betas[:,:, best_lambda_index]  # Coefficients for optimal lambda
    adaptivelasso_index = findall(!iszero,selected_coefficients[:,1])
end
    
    

###################################################################################
## USE SVD for collinearity
print(beta_dic)
#########
# Set random seed for reproducibility
Random.seed!(42)

# Generate a random 100 x 10 matrix
X = randn(50, 10)
true_beta = [repeat([2], 4); repeat([0], 6)] # Random values in [-10, 10]
#X[:, 2] = 2.0 * X[:, 6] - 0.5 * X[:, 9]
Y = X * true_beta + rand(Normal(0, 2),50)

@time begin
    SIC_tmp = GIC_Variable_Selection(X, Y, collect(1:10), Calculate_SIC, Calculate_SIC_short)
end

# Introduce collinearity: Make column 3 a linear combination of columns 1 and 2

# X[:, 3] = X[:, 1] + X[:, 2]
# X[:, 4] = X[:, 1] + X[:, 3]
# X = X[:,1:500]
# # Verify collinearity
# println("Collinearity check (should be close to zero):")
# println(norm(X[:, 3] - (2.0 * X[:, 1] - 0.5 * X[:, 2])))

function softmax(z)
    exp_z = exp.(z)  # Element-wise exponentiation
    return exp_z ./ sum(exp_z)  # Normalize by the sum of exponentials
end

X_sub = X[:,1:5]
X_sub = X
(T, K) = (size(X_sub, 1), size(X_sub, 2))
Inverse = inv(X_sub'*X_sub)
Hat_matrix = X_sub*Inverse*X_sub'

Y' * Hat_matrix* Y  - K*log(T)

Y' * softmax(Hat_matrix*Y) - K*log(T)/T


#U, D, V = svd(X; full = true)
U, D, V = svd(X)
U, D, V = svd(X[:,1:3])
U*U'
Hat_matrix != U*U'
# y  = U[:,1:4] * diagm(D[1:4]) * V[:,1:4]'
# print( U[:,1])
# V'
# U[:,1:4] * U[:,1:4]'*Y
# U * U'*Y
(T, K) = (size(U, 1), size(U, 2))

print(softmax(U'*Y))
print(U'*Y)
# print(U'*X * true_beta)
print(sum(softmax(U'*Y)))
print(sum(abs.(U'*Y))- K*log(T))
print(" ")
print(sum(abs.(U'*Y)))
print(" ")
print(sum(abs.(U'*X * true_beta)))



print(exp.(U'*Y))
print(log(sum(exp.(U'*Y .- K*log(T)))))
print(log(sum(exp.(U'*Y))))


###########################################################################
###########################################################################
# Data Simulation and Summarization
function simulate_data(m, P, k, random_index_true_columns, SNR, rho, Nsim)
    # Placeholder for your IC_main equivalent logic
    return DataFrame(SNR_l=SNR, Label=["AIC", "BIC", "Lasso"], RR_o=rand(3), RR_i=rand(3),
                     RTE=rand(3), PVE=rand(3), Over=rand(3), Under=rand(3), Correct=rand(3), Total=rand(3))
end

function generate_data(m, P, k, random_index_true_columns, SNR, rho)
    big_data = []
    for j in 1:10
        datalist = []
        for i in 1:length(SNR)
            push!(datalist, Comparison_Function(m, P, k, random_index_true_columns, SNR[i], rho, Nsim=j))
        end
        push!(big_data, vcat(datalist...))
    end
    return vcat(big_data...)
end

# Parameters
N, P, k = 500, 100, 16
random_index_true_columns = sort(sample(1:P, k, replace=false))
random_initial_columns = sort(sample(1:P, 30, replace=false))
# Randomly select `k` unique indices from `1:P` and sort them
random_index_true_columns = sort(sample(1:P, k, replace=false))
SNR = [0.05, 0.09, 0.14, 0.25, 0.42, 0.71, 1.22, 2.07, 3.52, 6.00]
rho = 0.0

Comparison_Function(N, P, k, random_index_true_columns, SNR[1], rho, Nsim=1)

Comparison_Function(N, P, k, random_true_columns, SNR, rho, Nsim)

whole_data = generate_data(N, P, k, random_index_true_columns, SNR, rho)
println(first(whole_data, 5))  # Preview the data

data_tmp = Data_Generation(m, P, k, random_index_true_columns, SNR[8], rho)

print(data_tmp[:beta_dic])

# Data Summarization
# Group and summarize
grouped_data = @chain whole_data begin
    groupby([:SNR_l, :Label])
    combine(:RR_o => median => :RR_o_mu,
            :RR_i => median => :RR_i_mu,
            :RTE => median => :RTE_mu,
            :PVE => median => :PVE_mu,
            :Over => median => :OVER_mu,
            :Under => median => :UNDER_mu,
            :Correct => median => :CORRECT_mu,
            :Total => median => :TOTAL_mu,
            :RR_o => std => :RR_o_sd,
            :RR_i => std => :RR_i_sd,
            :RTE => std => :RTE_sd,
            :PVE => std => :PVE_sd)
end

println(first(grouped_data, 5))  # Preview summarized data

# Visualization 
using Plots

colors = Dict("AIC" => :red, "BIC" => :green, "Lasso" => :blue)

# Set a plotting theme
theme(:default)

# Create Individual Plots 
function plot_metric(data, y_metric, y_label, colors)
    plot()
    grouped = groupby(data, :Label)
    for (label, group) in grouped
        x = group.SNR_l
        y = group[y_metric]
        plot!(x, y, label=label, color=colors[label], lw=2, legend=:bottomright)
    end
    xlabel!("Signal to Noise Ratio")
    ylabel!(y_label)
end

# Create plots
plot_metric(grouped_data, :RR_o_mu, "Out-of-Sample Relative Risk", colors)
plot_metric(grouped_data, :RR_i_mu, "In-Sample Relative Risk", colors)
plot_metric(grouped_data, :RTE_mu, "Relative Test Error", colors)
plot_metric(grouped_data, :PVE_mu, "Proportion of Variance Explained", colors)

# Create Multiple Plots 
using Plots.PlotMeasures

p1 = plot_metric(grouped_data, :RR_o_mu, "Out-of-Sample Relative Risk", colors)
p2 = plot_metric(grouped_data, :RR_i_mu, "In-Sample Relative Risk", colors)
p3 = plot_metric(grouped_data, :RTE_mu, "Relative Test Error", colors)
p4 = plot_metric(grouped_data, :PVE_mu, "Proportion of Variance Explained", colors)

plot(p1, p2, p3, p4, layout=(2, 2), title="Model Selection Metrics")

# Save Plots 
# savefig("/path/to/save/RR_F.png")

