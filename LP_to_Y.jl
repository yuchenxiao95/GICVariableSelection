using StatsBase
using Distributions

function LP_to_Y(X, true_beta; 
    family::String = "Normal", 
    n_trials::Union{Nothing, Int64} = nothing, 
    std::Union{Nothing, Float64} = nothing, 
    shape::Union{Nothing, Float64} = nothing,
    cov_matrix::Union{Nothing, Matrix{Float64}} = nothing,
    n_categories::Union{Nothing, Int64} = nothing)
"""
Simulate outcomes for different GLM families, including Multivariate Normal and Multinomial.

# Arguments
- `X::Matrix{Float64}`: Design matrix of size `(n_samples × n_features)`.
- `true_beta::Union{Vector{Float64}, Matrix{Float64}}`: 
  - For univariate families: Coefficient vector of size `(n_features,)`.
  - For `MultivariateNormal`: Coefficient matrix of size `(n_features × p)`.
  - For `Multinomial`: Coefficient matrix of size `(n_features × (n_categories-1))`.

# Keyword Arguments
- `family::String`: Distribution family. Options:
  - Univariate: `"Bernoulli"`, `"Binomial"`, `"Normal"`, `"Poisson"`, `"Gamma"`, `"Exponential"`
  - Multivariate: `"MultivariateNormal"`, `"Multinomial"`.
- `n_trials::Union{Nothing, Int64}`: Required for `"Binomial"` (number of trials).
- `std::Union{Nothing, Float64}`: Standard deviation for `"Normal"` (default: 1.0).
- `shape::Union{Nothing, Float64}`: Shape parameter for `"Gamma"` (default: 1.0).
- `cov_matrix::Union{Nothing, Matrix{Float64}}`: Required for `"MultivariateNormal"`.
- `n_categories::Union{Nothing, Int64}`: Required for `"Multinomial"` (number of classes).

# Returns
- `Union{Vector{Float64}, Matrix{Float64}, Vector{Int}}`: Simulated responses:
  - Univariate: Vector of size `(n_samples,)`.
  - `MultivariateNormal`: Matrix of size `(n_samples × p)`.
  - `Multinomial`: Vector of class labels `1:n_categories`.

"""
    # Calculate the linear predictor (X * true_beta)

    linear_predictor = X * true_beta

    if family == "Bernoulli"
        mu = 1.0 ./(1.0 .+ exp.(-linear_predictor))
        return Float64.(rand.(Bernoulli.(mu)))

    elseif family == "Binomial"
        mu = 1.0 ./(1.0 .+ exp.(-linear_predictor))
        if isnothing(n_trials)
        error("n_trials must be specified for Binomial family.")
        end
        return rand.(Binomial.(n_trials, mu))

    elseif family == "Normal"
        mu = linear_predictor
        if isnothing(std)
        std = 1.0
        end
        return mu .+ rand(Normal(0, std), size(mu))

    elseif family == "Poisson"
        max_value = 30.0
        linear_predictor = clamp.(linear_predictor, -max_value, max_value)
        mu = exp.(linear_predictor)
        return rand.(Poisson.(mu))

    elseif family == "Gamma"
        mu = exp.(linear_predictor)
        if isnothing(shape)
        shape = 1.0
        end
        return rand.(Gamma.(shape, mu))

    elseif family == "Exponential"
        max_value = 30.0
        linear_predictor = clamp.(linear_predictor, -max_value, max_value)
        mu = exp.(linear_predictor)
        return rand.(Exponential.(mu))

    elseif family == "MultivariateNormal"
        if isnothing(cov_matrix)
        error("cov_matrix must be specified for MultivariateNormal family.")
        end
        if !isposdef(cov_matrix)
        error("Covariance matrix must be positive definite.")
        end

        mu = linear_predictor
        n_samples, p = size(mu)
        samples = zeros(n_samples, p)

        for i in 1:n_samples
        mv_normal = MvNormal(mu[i, :], cov_matrix)
        samples[i, :] = rand(mv_normal)
        end
        return samples

    elseif family == "Multinomial"
        if isnothing(n_categories) || n_categories < 2
        error("n_categories must be specified and ≥2 for Multinomial family.")
        end

        n_samples = size(X, 1)
        y = zeros(Int, n_samples)

        # For multinomial, true_beta should be matrix of size (n_features × (n_categories-1))
        if size(true_beta, 2) != n_categories - 1
        error("For Multinomial family, true_beta must have size (n_features × (n_categories-1))")
        end

        for i in 1:n_samples
        # Calculate logits for each class
        logits = zeros(n_categories)
        logits[1:n_categories-1] = linear_predictor[i, :]
        logits[n_categories] = 0.0  # reference class

        # Softmax transformation
        max_logit = maximum(logits)
        probs = exp.(logits .- max_logit)
        probs ./= sum(probs)

        # Sample from multinomial
        y[i] = rand(Categorical(probs))
        end
        return y

    else
        error("Unsupported family: ", family)
    end
end
