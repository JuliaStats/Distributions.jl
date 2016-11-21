# Tests for DirichletMultinomial

using Distributions
using Base.Test

d1 = DirichletMultinomial(10, 5)
d2 = DirichletMultinomial(10, rand(5))
