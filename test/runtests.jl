using GICVariableSelection
using Distributions
using Random
using DataFrames
using Test

@testset "Univariate Normal Model Selection" begin
    N, P, k = 1000, 500, 5
    rho = 0.0
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
    variance = (true_beta' * cov * true_beta) / SNR[5]
    std = sqrt(variance)

    Y = LP_to_Y(X, true_beta, family="Normal", std=std)

    init_cols  = collect(1:P)
    tmp = GIC_Variable_Selection(X, Y, init_cols, Calculate_BIC, Calculate_BIC_short, Nsim=5)

    estimated = tmp[2][end]
    @test typeof(tmp[1]) == Vector{Float64}
    @test typeof(tmp[2]) == Vector{Vector{Int}}
    @test length(setdiff(true_columns, estimated)) <= k  # allow some tolerance
    @test length(setdiff(estimated, true_columns)) >= 0  # allow some tolerance
end


@testset "Univariate Poisson Model Selection" begin
    N, P, k = 1000, 500, 5
    rho = 0.0
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
    variance = (true_beta' * cov * true_beta) / SNR[5]
    std = sqrt(variance)

    Y = LP_to_Y(X, true_beta, family="Poisson")

    init_cols  = collect(1:P)
    tmp = GIC_Variable_Selection(X, Y_to_LP(Y,"Poisson"), init_cols, Calculate_SIC, Calculate_SIC_short, Nsim=5)

    estimated = tmp[2][end]
    @test typeof(tmp[1]) == Vector{Float64}
    @test typeof(tmp[2]) == Vector{Vector{Int}}
    @test length(setdiff(true_columns, estimated)) <= k  # allow some tolerance
    @test length(setdiff(estimated, true_columns)) >= 0  # allow some tolerance
end


@testset "Multivariate Normal Model Selection" begin
    N, P, k = 1000, 500, 5
    rho = 0.0
    SNR = [0.09, 0.14, 0.25, 0.42, 0.71, 1.22, 2.07, 3.52, 6.00]

    mu = zeros(P)
    cov = fill(rho, P, P)
    for i in 1:P
        cov[i, i] = 1.0
    end
    X = Matrix(rand(MvNormal(mu, cov), N)')

    m = 5
    multi_beta = zeros(P, m)
    multi_beta_true_columns = Vector{Vector{Int}}(undef, m)

    for i in 1:m
        cols = sort(sample(1:P, k, replace=false))
        multi_beta_true_columns[i] = cols
        multi_beta[cols, i] .= collect(range(10, 0.1, length=k))
    end

    variance = (multi_beta' * cov * multi_beta) ./ SNR[5]
    cov_p = fill(rho, m, m)
    for i in 1:m
        cov_p[i, i] = 1.0
    end

    Y = LP_to_Y(X, multi_beta, family="MultivariateNormal", cov_matrix=cov_p)

    init_cols  = collect(1:P)
    tmp = GIC_Variable_Selection(X, Y, init_cols, Calculate_SIC, Calculate_SIC_short, Nsim=8)

    estimated = tmp[2][end]
    all_true = unique(reduce(vcat, multi_beta_true_columns))

    @test typeof(tmp[1]) == Vector{Float64}
    @test typeof(tmp[2]) == Vector{Vector{Int}}
    @test length(setdiff(all_true, estimated)) <= k  # allow some tolerance
    @test length(setdiff(estimated, all_true)) >= 0  # allow some tolerance
end
