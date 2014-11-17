using Distributions
using Base.Test

# RealInterval
r = RealInterval(1.5, 4.0)
@test minimum(r) == 1.5
@test maximum(r) == 4.0

r = Distributions.make_support(DiscreteUniform, 1, 5)
@test isa(r, UnitRange)
@test first(r) == 1
@test last(r) == 5

r = Distributions.make_support(Uniform, 1.0, 5.0)
@test isa(r, RealInterval)
@test minimum(r) == 1.0
@test maximum(r) == 5.0
@test r == RealInterval(1.0, 5.0)

