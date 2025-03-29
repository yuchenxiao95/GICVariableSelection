using Distributions, LinearAlgebra

function Data_Generation(N, P, k, true_columns, SNR, rho)
    # Inputs:
    # N: Number of observations
    # P: Number of predictors
    # k: Number of true predictors
    # true_columns: Indices of true predictors
    # SNR: Signal-to-noise ratio
    # rho: Correlation between predictors
    # Parameters
    N, P, k = 1000, 500, 5
    rho = 0.4
    random_index_true_columns = sort(sample(1:P, k, replace=false))
    SNR = [0.09, 0.14, 0.25, 0.42, 0.71, 1.22, 2.07, 3.52, 6.00]
    true_columns = random_index_true_columns
    # Set the mean vector and covariance matrix
    mu = zeros(P)  # Mean vector of zeros
    cov = fill(rho, P, P)  # Correlation matrix with rho

    for i in 1:P
        cov[i, i] = 1.0  # Set diagonal to 1
    end
    # Generate the design matrix X from a multivariate normal distribution
    dist = MvNormal(mu, cov)
    X = Matrix(rand(dist, N)')


    # Set the true coefficients for univariate case 
    #coeff_true = [rand(Uniform(5,10),5); rand(Uniform(-10,-5),5); rand(Uniform(2,5),3); rand(Uniform(-5,-2),3)]  # Random values in [-10, 10]
    # true_beta = zeros(P)
    # coeff_true = [10 for _ in 1:10]
    # true_beta[true_columns] .= coeff_true  # Assign true coefficients to the specified columns
    #Compute the variance based on SNR
    # variance = (true_beta' * cov * true_beta) / SNR[5]
    # std = sqrt(variance)
    # beta_dic = Dict(true_columns .=> coeff_true)
    # Y = simulate_glm_outcomes(X, true_beta, family= "Normal", std = std)


    # Set the true coefficients for multivariate case 
    m = 5
    multi_beta = zeros(P, m)
    multi_beta_true_columns = Vector{Vector{Int}}(undef, m)
    # Assign non-zero coefficients for each response variable
    for i in 1:m  # Loop over columns (responses)
        # Randomly select k features to have non-zero coefficients
        true_columns = sort(sample(1:P, k, replace=false))
        multi_beta_true_columns[i] = true_columns 
        # Set the non-zero coefficients (e.g., all 2.0 for this example)
        #beta[1:k] = range(10, 0.5, length=k)
        multi_beta[true_columns, i] .=  collect(range(10, 0.1, length=k))
    end

    #Compute the variance based on SNR
    variance = (multi_beta' * cov * multi_beta) ./ SNR[5]
    std = sqrt(variance)

    cov_p = fill(rho, m, m)  # Correlation matrix with rho
    for i in 1:m
        cov_p[i, i] = 1.0  # Set diagonal to 1
    end

    Y = simulate_glm_outcomes(X, multi_beta, family= "MultivariateNormal", cov_matrix = cov_p)
    
 
 

    # Generate some example data
    n_samples = 100
    n_features = 5
    n_categories = 3
    X = randn(n_samples, n_features)
    true_beta = randn(n_features, n_categories-1)  # Note: K-1 columns

    # Simulate multinomial outcomes
    Y = simulate_glm_outcomes(X, true_beta, 
                            family="Multinomial",
                            n_categories=n_categories)

end


function simulate_glm_outcomes(X, true_beta; 
                                family::String = "Normal", 
                                n_trials::Union{Nothing, Int64} = nothing, 
                                std::Union{Nothing, Float64} = nothing, 
                                shape::Union{Nothing, Float64} = nothing,
                                cov_matrix::Union{Nothing, Matrix{Float64}} = nothing,
                                n_categories::Union{Nothing, Int64} = nothing)
    """
    Simulate outcomes for different GLM families, including Multivariate Normal and Multinomial.
    
    Args:
        X (Matrix{Float64}): Design matrix of size (n_samples x n_features).
        true_beta (Union{Vector{Float64}, Matrix{Float64}}): Coefficient vector or matrix.
            - For univariate families: vector of size (n_features,)
            - For MultivariateNormal: matrix of size (n_features x p)
            - For Multinomial: matrix of size (n_features x (n_categories-1))
        family (String): Distribution family. Supported values: 
            "Bernoulli", "Binomial", "Normal", "Poisson", "Gamma", 
            "Exponential", "MultivariateNormal", "Multinomial".
        n_trials (Union{Nothing, Int64}): Number of trials for Binomial family.
        std (Union{Nothing, Float64}): Standard deviation for Normal family.
        shape (Union{Nothing, Float64}): Shape parameter for Gamma family.
        cov_matrix (Union{Nothing, Matrix{Float64}}): Covariance matrix for MultivariateNormal.
        n_categories (Union{Nothing, Int64}): Number of categories for Multinomial family.
    
    Returns:
        Union{Vector{Float64}, Matrix{Float64}}: Simulated outcomes.
            - For univariate families: vector of size (n_samples,)
            - For MultivariateNormal: matrix of size (n_samples x p)
            - For Multinomial: vector of size (n_samples,) with class labels (1 to K)
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

# elseif family == "Categorical"
#     # For Categorical (binary case assumed): Inverse link function is the logistic function (logit)
#     mu = 1.0 ./ (1.0 .+ exp.(-linear_predictor))  # Logistic transformation to get probability
#     return [Float64.(rand(Categorical([p, 1.0 - p]))) for p in mu]   # Simulate from a binary categorical distribution
  
# elseif family == "Cox"
#     max_value = 30.0  # Adjust the threshold as needed to avoid overflow
#     linear_predictor = clamp.(linear_predictor, -max_value, max_value)
#     # For Cox: Hazard function is typically modeled with log(mu)
#     mu = exp.(linear_predictor)
#     return rand.(Exponential.(mu))  # Simulate using Exponential distribution for hazard rate


    
 

    # Return the data and parameters
    return Dict(
        :X => X,
        :Y => Y,
        :true_beta => true_beta,
        :beta_dic => beta_dic,
        :cov => cov,
        :variance => variance
    )
end
