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

function Y_to_LP(Y::Vector{Float64}, family::String;
                  shape::Union{Nothing, Float64} = nothing, 
                  n_trials::Union{Nothing, Int64} = nothing, 
                  std::Union{Nothing, Float64} = nothing)

    if family == "Binomial"
        if isnothing(n_trials)
            error("`n_trials` must be specified for Binomial family.")
        end
        p = Y ./ n_trials
        p_adjusted = copy(p)
        p_adjusted[p .== 0.0] .= 0.01  # avoid log(0)
        p_adjusted[p .== 1.0] .= 0.99  # avoid log(∞)
        return log.(p_adjusted ./ (1 .- p_adjusted))

    elseif family == "Bernoulli"
        Y_adjusted = copy(Y)
        Y_adjusted[Y .== 0.0] .= 0.01
        Y_adjusted[Y .== 1.0] .= 0.99
        return log.(Y_adjusted ./ (1 .- Y_adjusted))

    elseif family == "Normal"
        return Y

    elseif family == "Poisson"
        Y_adjusted = copy(Y)
        Y_adjusted[Y .== 0.0] .= 0.1
        return log.(Y_adjusted)

    elseif family == "Gamma" || family == "Exponential"
        Y_adjusted = copy(Y)
        Y_adjusted[Y .== 0.0] .= 0.1
        return log.(Y_adjusted)

    elseif family == "MultivariateNormal"
        return Y

    else
        error("Unsupported family: $family")
    end
end
