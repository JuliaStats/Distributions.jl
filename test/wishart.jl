# Tests on Wishart distributions

require("../src/Distributions.jl")

using Distributions
using Base.Test

V = [[2. 1.], [1. 2.]]
W = Wishart(3., V)

# logdet

@test_approx_eq expected_log_det(W) 1.9441809588650447

# entropy

@test_approx_eq entropy(W) 7.178942679971454
