using Distributions
using Base.Test

# asserts that the pdf function can handle larger n values (up to 1029 with this shape)
d = BetaBinomial(1029, 2, 2)

# generatd with srand(1); sample(1:1029, 3, replace=false, ordered=true)
random_indices = [244, 251, 416]
answers = Float64[0.001051332818986906, 0.0010718746188104874, 0.0014006967130291636]

@test isapprox(pdf(d)[random_indices], answers)
@test isapprox(pdf(d, random_indices[1]), answers[1])
@test isapprox(pdf(d, random_indices[2]), answers[2])
@test isapprox(pdf(d, random_indices[3]), answers[3])
