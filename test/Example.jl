import Pkg
Pkg.activate(".")

Pkg.rm("GICVariableSelection")  # optional, to remove completely
Pkg.add(url="https://github.com/yuchenxiao95/GICVariableSelection")

using GICVariableSelection, Plots, StatsBase, Distributions, DataFrames, LinearAlgebra


N, P, k = 3000, 300, 5
rho = 0.1
true_columns = sort(sample(1:P, k, replace=false))
SNR = [0.09, 0.14, 0.25, 0.42, 0.71, 1.22, 2.07, 3.52, 6.00]

mu = zeros(P)
cov = fill(rho, P, P)
for i in 1:P
cov[i, i] = 1.0
end
X = Matrix(rand(MvNormal(mu, cov), N)')

# Simulate multicolinearity
# Choose 5 of them to be replaced
num_replace = 3
replace_columns = sample(true_columns, num_replace; replace = false)
other_columns = setdiff(1:P, true_columns)

# Replace each selected column with a linear combination of 2–7 other columns
for col in replace_columns
    # Pick 2–7 other columns randomly (excluding the current one)
    other_cols = setdiff(other_columns, col)
    mix_cols = sample(other_cols, rand(2:7); replace = false)

    # Generate random coefficients for the linear combination
    coeffs = 2*rand(length(mix_cols))  # or normalize if needed

    # Replace the column with the linear combination
    X[:, col] = X[:, mix_cols] * coeffs
end

true_beta = zeros(P)
true_beta[true_columns] .= 1
variance = (true_beta' * cov * true_beta) / SNR[9]
std = sqrt(variance)

Y = LP_to_Y(X, true_beta, family="Normal", std=std)


init_cols  = collect(1:P)
@time begin 
tmp = GIC_Variable_Selection(X, Y, init_cols, Calculate_BIC, Calculate_BIC_short, Nsim=5)
end


tmpp = DataFrame(A = tmp[1], 
                    B = tmp[2])
Plots.plot(tmp[1])
#random_index_true_columns = unique(reduce(vcat, multi_beta_true_columns)) 
print(setdiff(tmp[2][end],true_columns))
print(setdiff(true_columns, tmp[2][end]))


estimate_beta = zeros(P)
IC, Inverse = Calculate_BIC(Y, X[:,tmp[2][end]])
################
U, S, V = svd(X[:,tmp[2][end]])
# Invert the squared singular values
S_inv2 = Diagonal(1 ./ (S .^ 2))
# Compute (X'X)^-1 from SVD: V * S^-2 * V'
Inverse = V * S_inv2 * V'
######################
print("SIC")
Beta_estimate(Y, X[:,tmp[2][end]], Inverse)
estimate_beta[tmp[2][end]] .= Beta_estimate(Y, X[:,tmp[2][end]], Inverse)
sum((estimate_beta.- true_beta) .* (estimate_beta.- true_beta)) / P



estimate_beta = zeros(P)
IC, Inverse = Calculate_BIC(Y, X[:,true_columns])
print("True")
Beta_estimate(Y, X[:,true_columns], Inverse)
estimate_beta[true_columns] .= Beta_estimate(Y, X[:,true_columns], Inverse)
sum((estimate_beta .- true_beta) .* (estimate_beta .- true_beta)) / P



estimate_beta = zeros(P)
IC, Inverse = Calculate_BIC(Y, X)
print("Full")
Beta_estimate(Y, X, Inverse)
estimate_beta .= Beta_estimate(Y, X, Inverse)
sum((estimate_beta .- true_beta) .* (estimate_beta .- true_beta)) / P


