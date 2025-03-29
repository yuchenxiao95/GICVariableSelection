using StatsBase
using Distributions

# Define the function to simulate outcomes for different families
function LP_to_Y(X, true_beta; 
    family::String = "Normal", 
    n_trials::Union{Nothing, Int64} = nothing, 
    std::Union{Nothing, Float64} = nothing, 
    shape::Union{Nothing, Float64} = nothing)
    # Calculate the linear predictor (X * beta)
    linear_predictor = X * true_beta

    # Compute the mean (mu) based on the family-specific inverse link function
    if family == "Bernoulli"
    # For Bernoulli: Inverse link function is the logistic function (logit)
    mu = 1.0 ./(1.0 .+ exp.(-linear_predictor))
    return Float64.(rand.(Bernoulli.(mu)))   # Bernoulli outcome (0 or 1)

    elseif family == "Binomial"
    # For Binomial: Inverse link function is the logistic function (logit)
    mu = 1.0 ./(1.0 .+ exp.(-linear_predictor))
    if isnothing(n_trials)
    error("n_trials must be specified for Binomial family.")
    end
    return rand.(Binomial.(n_trials, mu))  # Binomial outcome

    elseif family == "Normal"
    # For Normal: Inverse link function is the identity function
    mu = linear_predictor
    if isnothing(std)
    std = 1.0  # Default std deviation
    end
    return rand.(Normal.(mu, std))  # Simulate from Normal with mean mu and std

    elseif family == "Poisson"
    # Ensure the linear predictor is not too large by applying a max value
    max_value = 30.0  # Adjust the threshold as needed to avoid overflow
    linear_predictor = clamp.(linear_predictor, -max_value, max_value)
    # For Poisson: Inverse link function is exp (log link)
    mu = exp.(linear_predictor)
    return rand.(Poisson.(mu))  # Simulate from Poisson with mean mu

    elseif family == "Gamma"
    # For Gamma: Inverse link function is exp (log link)
    mu = exp.(linear_predictor)
    if isnothing(shape)
    shape = 1.0  # Default shape parameter
    end
    return rand.(Gamma.(shape, mu))  # Simulate from Gamma with shape and mean mu

    elseif family == "Exponential"
    max_value = 30.0  # Adjust the threshold as needed to avoid overflow
    linear_predictor = clamp.(linear_predictor, -max_value, max_value)
    # For Exponential: Inverse link function is exp (log link)
    mu = exp.(linear_predictor)
    return rand.(Exponential.(mu))  # Simulate from Exponential with rate=mu

    else
    error("Unsupported family: ", family)
    end
end
