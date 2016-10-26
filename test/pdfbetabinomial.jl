using Distributions
using Base.Test

d = BetaBinomial(3, 2, 2)
answers = [0.19999999999999998, 0.29999999999999977, 0.29999999999999977, 0.19999999999999998]

@test isapprox(pdf(d), answers)
@test isapprox(pdf(d, 0), answers[1])
@test isapprox(pdf(d, 1), answers[2])
@test isapprox(pdf(d, 2), answers[3])
@test isapprox(pdf(d, 3), answers[4])

d = BetaBinomial(1029, 2, 2)
# generatd with srand(1); sample(1:1029, 3, replace=false, ordered=true)
random_indices = [244, 251, 416]
answers = [0.001051332818986906, 0.0010718746188104872, 0.0014006967130291636]

@test isapprox(pdf(d)[random_indices], answers)
# 0-based indices, subtract 1
@test isapprox(pdf(d, random_indices[1] - 1), answers[1])
@test isapprox(pdf(d, random_indices[2] - 1), answers[2])
@test isapprox(pdf(d, random_indices[3] - 1), answers[3])
