@testset "laplace.jl" begin
    # affine transformations
    test_cgf(Laplace(1, 1), (0.99, -0.99, 1.0f-2, -1.0f-5))
    test_affine_transformations(Laplace, randn(), randn()^2)
end
