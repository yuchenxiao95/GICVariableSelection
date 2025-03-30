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
    rho = 0.0
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

    Y = LP_to_Y(X, multi_beta, family= "MultivariateNormal", cov_matrix = cov_p)
    
 
 

    # # Generate some example data
    # n_samples = 100
    # n_features = 5
    # n_categories = 3
    # X = randn(n_samples, n_features)
    # true_beta = randn(n_features, n_categories-1)  # Note: K-1 columns

    # # Simulate multinomial outcomes
    # Y = simulate_glm_outcomes(X, true_beta, 
    #                         family="Multinomial",
    #                         n_categories=n_categories)

end


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
