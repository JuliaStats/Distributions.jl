# Tests for Logistic

using Distributions
using Test

d = Logistic()

# PDF
@test pdf(d, -Inf) ≈ pdf(d, Inf)
