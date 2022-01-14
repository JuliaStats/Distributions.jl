@testset "laplace.jl" begin
    # affine transformations
    test_affine_transformations(Laplace, randn(), randn()^2)
end
