@testset "Gumbel" begin
    @testset "eltype" begin
        @test @test_deprecated(eltype(Gumbel())) === Float64
        @test @test_deprecated(eltype(Gumbel(1f0))) === Float32
        @test @test_deprecated(eltype(Gumbel{Int}(0, 1))) === Float64
    end

    @testset "rand" begin
        d = Gumbel(rand(), rand())

        samples = [rand(d) for _ in 1:10_000]
        @test mean(samples) ≈ mean(d) rtol=5e-2
        @test std(samples) ≈ std(d) rtol=5e-2

        samples = rand(d, 10_000)
        @test mean(samples) ≈ mean(d) rtol=5e-2
        @test std(samples) ≈ std(d) rtol=5e-2

        d = Gumbel{Int}(0, 1)
        @test rand(d) isa Float64
        @test rand(d, 10) isa Vector{Float64}
        @test rand(d, (3, 2)) isa Matrix{Float64}
    end
end
