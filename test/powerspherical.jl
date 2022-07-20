@testset "Testing argument promotions" begin
    d = PowerSpherical(Int[1, 0], Float32(5))
    @test d isa PowerSpherical{Float32}
    d = PowerSpherical(Int[1, 0], Float64(5))
    @test d isa PowerSpherical{Float64}
    d = PowerSpherical(Float64[1, 0], 5)
    @test d isa PowerSpherical{Float64}
    d = PowerSpherical(Float64[1, 0], Float32(5))
    @test d isa PowerSpherical{Float64}
end