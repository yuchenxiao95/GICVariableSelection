using LinearAlgebra, Statistics, StatsBase
using DataFrames, StatsModels, GLM
import GLM: LogitLink, LogLink, IdentityLink, InverseLink
using Distributions

# ─────────────────────────────────────────────────────────────────────────────
# 1 · LP_to_Y  — simulate responses given a linear predictor
# ─────────────────────────────────────────────────────────────────────────────
"""
    LP_to_Y(X, β; family="Normal", …)

Given a design matrix `X` (n×p) and coefficient vector or matrix `β`,
simulate response `Y` according to the requested `family`.

For `family="Multinomial"` supply a coefficient matrix of size
`p × (K-1)`; pass `n_categories=K`.  The last class is the baseline.
"""
function LP_to_Y(X, β;
        family::AbstractString         = "Normal",
        n_trials::Union{Nothing,Int}   = nothing,
        std::Union{Nothing,Float64}    = nothing,
        shape::Union{Nothing,Float64}  = nothing,
        cov_matrix::Union{Nothing,AbstractMatrix} = nothing,
        n_categories::Union{Nothing,Int}= nothing)

    η = X * β
    n = size(X,1)

    if family == "Bernoulli"
        p = @. 1 / (1 + exp(-η))
        return rand.(Bernoulli.(p))

    elseif family == "Binomial"
        n_trials === nothing && error("n_trials required")
        p = @. 1 / (1 + exp(-η))
        return rand.(Binomial.(n_trials, p))

    elseif family == "Normal"
        σ = isnothing(std) ? 1.0 : std
        return η .+ σ .* randn(n)

    elseif family == "Poisson"
        μ = @. exp(clamp(η, -30, 30))
        return rand.(Poisson.(μ))

    elseif family in ("Gamma", "Exponential")
        κ = isnothing(shape) ? 1.0 : shape
        μ = @. exp(η)
        θ = μ ./ κ                       # scale so that mean = κθ
        return rand.(Gamma.(κ, θ))

    elseif family == "MultivariateNormal"
        cov_matrix === nothing && error("cov_matrix required")
        isposdef(cov_matrix) || error("cov_matrix not positive-definite")
        Y = similar(η)
        @inbounds for i in 1:n
            Y[i,:] = rand(MvNormal(vec(η[i,:]), cov_matrix))
        end
        return Y

    elseif family == "Multinomial"
        K = isnothing(n_categories) ? size(β,2)+1 : n_categories
        size(β,2) == K-1 || error("β must be p × (K-1)")
        P = hcat(η, zeros(n))            # add baseline logit
        P .-= maximum(P, dims=2)         # soft-max stability
        P  = exp.(P);   P ./= sum(P, dims=2)
        return [rand(Categorical(Vector(view(P, i, :)))) for i in 1:n]

    else
        error("Unsupported family $family")
    end
end