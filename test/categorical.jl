using Distributions
using Base.Test

d = Categorical([0.25, 0.5, 0.25])
d = Categorical(3)
d = Categorical([0.25, 0.5, 0.25])

@test !insupport(d, 0)
@test insupport(d, 1)
@test insupport(d, 2)
@test insupport(d, 3)
@test !insupport(d, 4)

@test logpmf(d, 1) == log(0.25)
@test pmf(d, 1) == 0.25

@test logpmf(d, 2) == log(0.5)
@test pmf(d, 2) == 0.5

@test logpmf(d, 0) == -Inf
@test pmf(d, 0) == 0.0

@test 1.0 <= rand(d) <= 3.0

A = Array(Int, 10)
rand!(d, A)
@test 1.0 <= mean(A) <= 3.0

# Examples of sample()
a = [1, 6, 19]
p = rand(Dirichlet(3))
x = sample(a, p)
@test x == 1 || x == 6 || x == 19

a = 19.0 * [1.0, 0.0]
x = sample(a)
@test x == 0.0 || x == 19.0

