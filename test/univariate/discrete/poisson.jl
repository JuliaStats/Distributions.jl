@testset "Poisson quantile (#1358)" begin
    d = Poisson(1)
    @test quantile(d, 0.2) isa Int
    @test cquantile(d, 0.4) isa Int
    @test invlogcdf(d, log(0.2)) isa Int
    @test invlogccdf(d, log(0.6)) isa Int
end
