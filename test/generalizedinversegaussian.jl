#using Distributions
#using SpecialFunctions
using Test

#include("../src/utils.jl")
#include("../src/univariates.jl")
#include("../src/univariate/continuous/generalizedinversegaussian.jl")
include("../src/Distributions.jl")

d = Distributions.GeneralizedInverseGaussian(1,2,1)

# x, pdf
pdfs = [
    0.0000001 0.000000000;
    0.5000001 0.118601367;
    1.0000001 0.251079021;
    1.5000001 0.272898789;
    2.0000001 0.251079002;
    2.5000001 0.216105698;
    3.0000001 0.179905964;
    3.5000001 0.146944261;
    4.0000001 0.118601320;
    4.5000001 0.094968513;
    5.0000001 0.075623540;
    5.5000001 0.059976297;
    6.0000001 0.047422697;
    6.5000001 0.037409379;
    7.0000001 0.029456378;
    7.5000001 0.023160177;
    8.0000001 0.018188101;
    8.5000001 0.014269445;
    9.0000001 0.011185927;
    9.5000001 0.008762703
]

for i in 1:size(pdfs,1)
    @test round(Distributions.pdf(d,pdfs[i,1]),digits=9) ≈ pdfs[i,2]
end