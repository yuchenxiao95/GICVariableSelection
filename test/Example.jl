import Pkg
Pkg.activate(".")

Pkg.rm("GICVariableSelection")  # optional, to remove completely
Pkg.add(url="https://github.com/yuchenxiao95/GICVariableSelection")

using GICVariableSelection, Plots, StatsBase, Distributions, DataFrames


N, P, k = 3000, 100, 5
rho = 0.6
true_columns = sort(sample(1:P, k, replace=false))
SNR = [0.09, 0.14, 0.25, 0.42, 0.71, 1.22, 2.07, 3.52, 6.00]

mu = zeros(P)
cov = fill(rho, P, P)
for i in 1:P
cov[i, i] = 1.0
end
X = Matrix(rand(MvNormal(mu, cov), N)')

true_beta = zeros(P)
true_beta[true_columns] .= 2
variance = (true_beta' * cov * true_beta) / SNR[4]
std = sqrt(variance)

Y = LP_to_Y(X, true_beta, family="Normal", std=std)

init_cols  = collect(1:P)
@time begin 
tmp = GIC_Variable_Selection(X, Y, init_cols, Calculate_ICOMP, Calculate_ICOMP_short, Nsim=10)
end

tmpp = DataFrame(A = tmp[1], 
                    B = tmp[2])
Plots.plot(tmp[1])
#random_index_true_columns = unique(reduce(vcat, multi_beta_true_columns)) 
#Calculate_BIC(Y_to_lp(Y, "Bernoulli"), X[:,true_columns])
print(setdiff(tmp[2][end],true_columns))
print(setdiff(true_columns, tmp[2][end]))

IC, Inverse = Calculate_ICOMP(Y, X[:,tmp[2][end]])


Beta_estimate(Y, X[:,tmp[2][end]], Inverse)





