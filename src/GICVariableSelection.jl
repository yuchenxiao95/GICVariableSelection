# This MUST be the first line
__precompile__(true)

module GICVariableSelection

# Standard libraries (NO need to declare in Project.toml)
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Distributed

# Regular packages
using CSV
using DataFrames
using Distributions
using IJulia
using LogExpFunctions
using Plots
using StatsBase
using StatsPlots

# Include your files
include("GIC_Calculation.jl")
include("GIC_Model_Selection.jl")
include("GIC_Model_Selection_Boltzmann.jl")
include("Beta_estimate.jl")
include("LP_to_Y.jl")
include("Y_to_lp.jl")

# Export your public API
export Calculate_AIC, Calculate_SIC, Calculate_BIC,
GIC_Variable_Selection, GIC_Variable_Selection_Boltzmann,
Beta_estimate, Y_to_lp, LP_to_Y

end # module

