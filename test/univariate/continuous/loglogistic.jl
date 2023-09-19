using Distributions
using Test

@testset "LogLogistic" begin 

    @test round(pdf(LogLogistic(), -1), digits=6) == 0.0
    @test round(pdf(LogLogistic(), 1), digits=6) == 0.25
    @test round(pdf(LogLogistic(2), 2), digits=6) == 0.125
    @test round(pdf(LogLogistic(2,2), 1), digits=6) == 0.32

    @test round(cdf(LogLogistic(), -1), digits=6) == 0.0
    @test round(cdf(LogLogistic(), 1), digits=6) == 0.5
    @test round(cdf(LogLogistic(2), 3), digits=6) == 0.6
    @test round(cdf(LogLogistic(2,2), 4), digits=6) == 0.8

    @test round(ccdf(LogLogistic(2), 3), digits=6) == 0.4
    @test round(ccdf(LogLogistic(2,2), 4), digits=6) == 0.2

    @test round(logpdf(LogLogistic(), -1), digits=6) == -Inf
    @test round(logpdf(LogLogistic(), 1), digits=6) == -1.386294

    @test round(logcdf(LogLogistic(2,2), 4), digits=6) == -0.223144
    @test round(logccdf(LogLogistic(2,2), 4), digits=6) == -1.609438 

end