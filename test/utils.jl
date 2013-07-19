using Distributions
using Base.Test

const randi = Distributions.randi
n = 1_000_000

x = Int[randi(10) for i = 1:n]
@test min(x) == 1
@test max(x) == 10

x = Int[randi(3, 12) for i = 1:n]
@test min(x) == 3
@test max(x) == 12
