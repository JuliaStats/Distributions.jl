using Distributions
using Base.Test

# asserts that the pdf function can handle n values up to 1029 without overflowing the precision and rounding to inf
d = BetaBinomial(1029, 2, 2)
_pdf = pdf(d)
# generatd with srand(1); sample(1:1029, 3, replace=false, ordered=true)
random_indices = [244, 251, 416]
answers = Float64[0.001051332818986906, 0.0010718746188104874, 0.0014006967130291636]
@test isapprox(_pdf[random_indices], answers)
