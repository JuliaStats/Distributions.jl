using Test
using Distributions

@testset "Truncated normal, mean and variance" begin
    d = Distributions.TruncatedNormal(0,1,100,115); @test mean(d) ≈ 100.00999800099926070518490239457545847490332879043
    d = Distributions.TruncatedNormal(0,1,50,70); @test var(d) ≈ 0.00039904318680389954790992722653605933053648912703600
    d = Distributions.TruncatedNormal(-2,3,50,70); @test mean(d) ≈ 50.171943499898757645751683644632860837133138152489
    d = Distributions.TruncatedNormal(-2,3,50,70); @test var(d) ≈ 0.029373438107168350377591231295634273607812172191712
end
