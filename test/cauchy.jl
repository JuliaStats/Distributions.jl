@testset "cauchy.jl" begin
    # affine transformations
    test_affine_transformations(Cauchy, randn(), randn()^2)
end
