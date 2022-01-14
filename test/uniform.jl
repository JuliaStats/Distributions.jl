@testset "uniform.jl" begin
    # affine transformations
    test_affine_transformations(Uniform, rand(), 4 + rand())
end
