using Distributions, Test, Random, LinearAlgebra, FillArrays
using Distributions: Product

@testset "Testing normal product distributions" begin
    Random.seed!(123456)
    N = 11
    # Construct independent distributions and `Product` distribution from these.
    μ = randn(N)
    ds = Normal.(μ, 1.0)
    x = rand.(ds)
    d_product = product_distribution(ds)
    @test d_product isa MvNormal
    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test eltype(d_product) === eltype(ds[1])
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    @test mean(d_product) == mean.(ds)
    @test var(d_product) == var.(ds)
    @test cov(d_product) == Diagonal(var.(ds))
    @test entropy(d_product) ≈ sum(entropy.(ds))

    y = rand(d_product)
    @test y isa typeof(x)
    @test length(y) == N
end

@testset "Testing generic product distributions" begin
    Random.seed!(123456)
    N = 11
    # Construct independent distributions and `Product` distribution from these.
    ubound = rand(N)
    ds = Uniform.(-ubound, ubound)
    x = rand.(ds)
    d_product = product_distribution(ds)
    @test d_product isa Product
    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test eltype(d_product) === eltype(ds[1])
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    @test mean(d_product) == mean.(ds)
    @test var(d_product) == var.(ds)
    @test cov(d_product) == Diagonal(var.(ds))
    @test entropy(d_product) == sum(entropy.(ds))
    @test insupport(d_product, ubound) == true
    @test insupport(d_product, ubound .+ 1) == false
    @test minimum(d_product) == -ubound
    @test maximum(d_product) == ubound
    @test extrema(d_product) == (-ubound, ubound)
    @test isless(extrema(d_product)...)

    y = rand(d_product)
    @test y isa typeof(x)
    @test length(y) == N
end

@testset "Testing discrete non-parametric product distribution" begin
    Random.seed!(123456)
    N = 11

    for a in ([0, 1], [-0.5, 0.5])
        # Construct independent distributions and `Product` distribution from these.
        support = fill(a, N)
        ds = DiscreteNonParametric.(support, Ref([0.5, 0.5]))
        x = rand.(ds)
        d_product = product_distribution(ds)
        @test d_product isa Product
        # Check that methods for `Product` are consistent.
        @test length(d_product) == length(ds)
        @test eltype(d_product) === eltype(ds[1])
        @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
        @test mean(d_product) == mean.(ds)
        @test var(d_product) == var.(ds)
        @test cov(d_product) == Diagonal(var.(ds))
        @test entropy(d_product) == sum(entropy.(ds))
        @test insupport(d_product, fill(a[2], N)) == true
        @test insupport(d_product, fill(a[2] + 1, N)) == false

        y = rand(d_product)
        @test y isa typeof(x)
        @test length(y) == N
    end
end

@testset "Testing iid product distributions" begin
    Random.seed!(123456)
    N = 11
    d = Product(Fill(Laplace(0.0, 2.3), N))
    @test N == length(unique(rand(d)));
    @test mean(d) === Fill(0.0, N)
    @test cov(d) === Diagonal(Fill(var(Laplace(0.0, 2.3)), N))
end

@testset "Testing non-independent product distributions" begin
    Random.seed!(123456)

    # Construct two non-isotropic Gaussians and some iid Laplaces
    N1, N2, N3 = 2, 3, 6
    N = N1 + N2 + N3

    μ1 = Fill(0, N1)
    Σ1 = sum(v -> v*v', eachcol(randn(N1, N1)))
    d1 = MvNormal(μ1, Σ1)
    
    μ2 = randn(N2)
    Σ2 = sum(v -> v*v', eachcol(randn(N2, N2)))
    d2 = MvNormal(μ2, Σ2)
    
    μ3 = randn(N3)
    b3 = randn(N3) .^ 2
    d3 = Laplace.(μ3, b3)

    d_product = Product([d1; d2; d3])

    _diagv(A) = [A[i,i] for i in 1:size(A, 1)]
    
    # check summary statistics
    @test length(d_product) == N
    @test mean(d_product) ≈ [μ1; μ2; μ3]
    @test var(d_product) ≈ [_diagv(Σ1); _diagv(Σ2); var.(d3)]
    @test var(d_product) ≈ _diagv(cov(d_product))

    # check additive properties
    x1, x2, x3 = randn(N1), randn(N2), randn(N3)
    x = [x1; x2; x3]

    @test insupport(d_product, x)
    @test logpdf(d_product, x) ≈ sum(logpdf.([d1, d2], [x1, x2])) +
        sum(logpdf.(d3, x3))
    @test entropy(d_product) ≈ sum(entropy, [d1; d2; d3])

    # check sampling
    y = rand(d_product)
    @test y isa Vector{eltype(d_product)}
    @test length(y) == N
end

@testset "Testing independent discrete distributions" begin
    Random.seed!(123456)

    d1 = Multinomial(10, [0.25, 0.25, 0.5])
    d2 = Geometric(0.5)
    N = 4

    d_product = Product([d1, d2])

    @test !insupport(d_product, Fill(0.5, N))
    @test length(d_product) == N
    
    # check sampling
    y = rand(d_product)
    @test y isa Vector{eltype(d_product)}
    @test length(y) == N
    
end

