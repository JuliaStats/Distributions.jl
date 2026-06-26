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

    @testset "global sparsity traces through relational (named-tuple) checks" begin
        # `@check_args` names the parameters of a relational check (e.g. `a < b`) with a named tuple
        # such as `(; a, b)`; tracing must skip the check yet still recover the true dependencies.
        detector = TracerSparsityDetector()

        @test jacobian_sparsity(x -> mean(Uniform(x[1], x[2])), [0.0, 1.0], detector) ==
            [true true]
        @test jacobian_sparsity(x -> minimum(Uniform(x[1], x[2])), [0.0, 1.0], detector) ==
            [true false]
        @test jacobian_sparsity(x -> maximum(Uniform(x[1], x[2])), [0.0, 1.0], detector) ==
            [false true]

        @test jacobian_sparsity(x -> mean(Arcsine(x[1], x[2])), [0.0, 1.0], detector) ==
            [true true]
        @test jacobian_sparsity(x -> minimum(Arcsine(x[1], x[2])), [0.0, 1.0], detector) ==
            [true false]
        @test jacobian_sparsity(x -> maximum(Arcsine(x[1], x[2])), [0.0, 1.0], detector) ==
            [false true]

        @test jacobian_sparsity(x -> minimum(LogUniform(x[1], x[2])), [1.0, 2.0], detector) ==
            [true false]
        @test jacobian_sparsity(x -> maximum(LogUniform(x[1], x[2])), [1.0, 2.0], detector) ==
            [false true]

        @test jacobian_sparsity(
            x -> minimum(censored(Normal(0.0, 1.0); lower=x[1], upper=x[2])), [-1.0, 1.0], detector
        ) == [true false]
        @test jacobian_sparsity(
            x -> maximum(censored(Normal(0.0, 1.0); lower=x[1], upper=x[2])), [-1.0, 1.0], detector
        ) == [false true]

        @test jacobian_sparsity(
            x -> mean(TriangularDist(x[1], x[2], x[3])), [0.0, 2.0, 1.0], detector
        ) == [true true true]
        @test jacobian_sparsity(
            x -> minimum(TriangularDist(x[1], x[2], x[3])), [0.0, 2.0, 1.0], detector
        ) == [true false false]
        @test jacobian_sparsity(
            x -> maximum(TriangularDist(x[1], x[2], x[3])), [0.0, 2.0, 1.0], detector
        ) == [false true false]
        @test jacobian_sparsity(
            x -> mode(TriangularDist(x[1], x[2], x[3])), [0.0, 2.0, 1.0], detector
        ) == [false false true]

        # `DiscreteUniform` (`(; a, b), a ≤ b`) and `OrderStatistic` (`(; rank, n), 1 ≤ rank ≤ n`)
        # also use the named-tuple form, but their checked parameters are integer-valued (`::Int`
        # fields / arguments) and can never be primal-free tracers, so global gradient tracing does
        # not apply to them; the recursion handles their named tuples consistently regardless.
    end

    @testset "local sparsity detection still validates arguments" begin
        detector = TracerLocalSparsityDetector()
        @test jacobian_sparsity(x -> mean(Normal(x[1], x[2])), [1.0, 2.0], detector) ==
            [true false]
        # The primal is available, so the `σ ≥ 0` check is enforced...
        @test_throws Exception jacobian_sparsity(
            x -> std(Normal(x[1], x[2])),
            [1.0, -1.0],
            detector,
        )
        # ...unless `check_args=false` skips it, so even an infeasible `σ` traces.
        @test jacobian_sparsity(
            x -> std(Normal(x[1], x[2]; check_args=false)),
            [1.0, -1.0],
            detector,
        ) == [false true]
        # Relational checks are likewise enforced: a valid `a < b` traces, an invalid one throws.
        @test jacobian_sparsity(x -> mean(Uniform(x[1], x[2])), [0.0, 1.0], detector) ==
            [true true]
        @test_throws Exception jacobian_sparsity(
            x -> mean(Uniform(x[1], x[2])),
            [1.0, 0.0],
            detector,
        )
        # `check_args=false` skips the check, so even infeasible values trace despite the primal.
        @test jacobian_sparsity(
            x -> mean(Uniform(x[1], x[2]; check_args=false)),
            [1.0, 0.0],
            detector,
        ) == [true true]
    end
end
