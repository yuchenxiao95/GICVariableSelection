using StatsBase
using Distributions

"""
    Y_to_LP(Y, family; shape=nothing, n_trials=nothing, std=nothing)

Transform observed responses `Y` into linear predictors using canonical link functions for various GLM families.

# Arguments
- `Y::Vector{Float64}`: Observed response vector.
- `family::String`: Distribution family. Supported options:
  - `"Bernoulli"`: Binary outcomes (0/1)
  - `"Binomial"`: Count outcomes with trials
  - `"Normal"`: Continuous outcomes
  - `"Poisson"`: Count outcomes
  - `"Gamma"`: Positive continuous outcomes
  - `"Exponential"`: Positive continuous outcomes
  - `"MultivariateNormal"`: Treated as identity

# Keyword Arguments
- `shape::Union{Nothing, Float64}`: Shape parameter for Gamma family (default = 1.0).
- `n_trials::Union{Nothing, Int64}`: Required for Binomial family (number of trials).
- `std::Union{Nothing, Float64}`: Unused placeholder for interface compatibility.

# Returns
- `Vector{Float64}`: Approximated linear predictors (η = g(μ)).
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Y_to_LP  — map Y back to canonical linear predictor
# ─────────────────────────────────────────────────────────────────────────────
function Y_to_LP(
        Y::AbstractVector,
        family::AbstractString;
        shape::Union{Nothing,Float64}    = nothing,
        n_trials::Union{Nothing,Int}     = nothing,
        n_categories::Union{Nothing,Int} = nothing)

    if family == "Binomial"
        n_trials === nothing && error("n_trials required")
        p = clamp.(Float64.(Y) ./ n_trials, 0.01, 0.99)
        return log.(p ./ (1 .- p))

    elseif family == "Bernoulli"
        p = clamp.(Float64.(Y), 0.01, 0.99)
        return log.(p ./ (1 .- p))

    elseif family == "Normal" || family == "MultivariateNormal"
        return Float64.(Y)

    elseif family == "Poisson"
        return log.(max.(Float64.(Y), 0.1))

    elseif family in ("Gamma", "Exponential")
        return log.(max.(Float64.(Y), 0.1))

    elseif family == "Multinomial"
        K = isnothing(n_categories) ? maximum(Y) : n_categories
        K ≥ 2 || error("n_categories must be ≥2")
        n = length(Y)
        base = 0.01 / (K-1)
        P   = fill(base, n, K)
        @inbounds for i in 1:n
            P[i, Y[i]] = 0.99
        end
        return log.(P[:,1:K-1] ./ P[:,K])          # n × (K-1)

    else
        error("Unsupported family $family")
    end
end
