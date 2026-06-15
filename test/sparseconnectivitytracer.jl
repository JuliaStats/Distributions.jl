using Distributions
using SparseConnectivityTracer
using Test

@testset "SparseConnectivityTracer" begin
    @testset "global gradient sparsity traces through construction" begin
        detector = TracerSparsityDetector()
        # `mean(Normal(μ, σ)) == μ` depends only on the first argument, `std` only on the second.
        @test jacobian_sparsity(x -> mean(Normal(x[1], x[2])), [1.0, 2.0], detector) ==
            [true false]
        @test jacobian_sparsity(x -> std(Normal(x[1], x[2])), [1.0, 2.0], detector) ==
            [false true]
    end

    @testset "global hessian sparsity traces through construction" begin
        detector = TracerSparsityDetector()
        # Two independent `Normal`s. Within each, `mean(d) * std(d) == μσ` couples that
        # distribution's own (μ, σ); the two distributions do not interact. So the Hessian is
        # block off-diagonal.
        f =
            x ->
                mean(Normal(x[1], x[2])) * std(Normal(x[1], x[2])) +
                mean(Normal(x[3], x[4])) * std(Normal(x[3], x[4]))
        @test hessian_sparsity(f, [1.0, 2.0, 3.0, 4.0], detector) == [
            false true false false
            true false false false
            false false false true
            false false true false
        ]
    end

    @testset "local sparsity detection still validates arguments" begin
        detector = TracerLocalSparsityDetector()
        @test jacobian_sparsity(x -> mean(Normal(x[1], x[2])), [1.0, 2.0], detector) ==
            [true false]
        # The primal is available, so the `σ ≥ 0` check is enforced.
        @test_throws Exception jacobian_sparsity(
            x -> std(Normal(x[1], x[2])),
            [1.0, -1.0],
            detector,
        )
    end
end
