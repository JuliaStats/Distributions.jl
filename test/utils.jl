using Distributions
using Base.Test

const randi = Distributions.randi
n = 1_000_000

x = Int[randi(10) for i = 1:n]
@test minimum(x) == 1
@test maximum(x) == 10

x = Int[randi(3, 12) for i = 1:n]
@test minimum(x) == 3
@test maximum(x) == 12
