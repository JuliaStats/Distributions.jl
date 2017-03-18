# Tests for GeneralizedInverseGaussian

using Distributions
using Base.Test

srand(123)

# Test Constructors
d = GeneralizedInverseGaussian(1, 1, 1)
d2 = GeneralizedInverseGaussian(3.0, 2, -0.5)
@test typeof(d) == typeof(d2)

p = randn()
d = GeneralizedInverseGaussian(10, 5, p)

# test parameters
@test params(d) == (10.0, 5.0, p)
