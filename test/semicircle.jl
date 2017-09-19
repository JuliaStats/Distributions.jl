using Distributions
using Base.Test

d = Semicircle(2.0)

@test pdf(d, 0)  == .31830988618379067153776752674
@test pdf(d, -2) == .0
@test pdf(d, 2)  == .0
