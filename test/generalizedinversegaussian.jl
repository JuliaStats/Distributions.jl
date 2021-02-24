using Distributions
using Test,Random, Statistics
using Plots


d = Distributions.GeneralizedInverseGaussian(1,2,1)

# x, pdf
pdfs = [
    0.0000001 0.000000000 -1.000000e7;
    0.5000001 0.118601367 -2.131987;
    1.0000001 0.251079021 -1.381988;
    1.5000001 0.272898789 -1.298654;
    2.0000001 0.251079002 -1.381988;
    2.5000001 0.216105698 -1.531988;
    3.0000001 0.179905964 -1.715321;
    3.5000001 0.146944261 -1.917702;
    4.0000001 0.118601320 -2.131988;
    4.5000001 0.094968513 -2.354210;
    5.0000001 0.075623540 -2.581988;
    5.5000001 0.059976297 -2.813806;
    6.0000001 0.047422697 -3.048654;
    6.5000001 0.037409379 -3.285834;
    7.0000001 0.029456378 -3.524845;
    7.5000001 0.023160177 -3.765321;
    8.0000001 0.018188101 -4.006988;
    8.5000001 0.014269445 -4.249635;
    9.0000001 0.011185927 -4.493099;
    9.5000001 0.008762703 -4.737251
]

@testset "pdf_cdf" begin
    for i in 1:size(pdfs,1)
        @test round(Distributions.pdf(d,pdfs[i,1]),digits=9) ≈ pdfs[i,2]
        @test round(Distributions.logpdf(d,pdfs[i,1]),digits=6) ≈ pdfs[i,3]
    end
    @test_throws MethodError Distributions.cdf(d,1) 
end

@testset "statistics" begin
    @test Distributions.mean(d) ≈ 3.076386787
    @test Distributions.mode(d) ≈ sqrt(2)
    @test Distributions.var(d) ≈ 4.841391484
end

