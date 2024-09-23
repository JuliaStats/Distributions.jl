# TODO: Remove when `MatrixReshaped` is removed
using Distributions, Test, Random, LinearAlgebra

rng = MersenneTwister(123456)

if VERSION >= v"1.6.0-DEV.254"
    _redirect_stderr(f, ::Base.DevNull) = redirect_stderr(f, devnull)
else
    function _redirect_stderr(f, ::Base.DevNull)
        nulldev = @static Sys.iswindows() ? "NUL" : "/dev/null"
        open(nulldev, "w") do io
            redirect_stderr(f, io)
        end
    end
end

function test_matrixreshaped(rng, d1, sizes)
    @testset "MatrixReshaped $(nameof(typeof(d1))) tests" begin
        x1 = rand(rng, d1)
        d1s = [@test_deprecated(MatrixReshaped(d1, s...)) for s in sizes]

        @testset "MatrixReshaped constructor" begin
            for d in d1s
                @test d isa MatrixReshaped
            end
        end
        @testset "MatrixReshaped constructor errors" begin
            @test_deprecated(@test_throws ArgumentError MatrixReshaped(d1, length(d1), 2))
            @test_deprecated(@test_throws ArgumentError MatrixReshaped(d1, length(d1)))
            @test_deprecated(@test_throws ArgumentError MatrixReshaped(d1, -length(d1), -1))
        end
        @testset "MatrixReshaped size" begin
            for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
                @test size(d) == s
            end
        end
        @testset "MatrixReshaped length" begin
            for d in d1s
                @test length(d) == length(d1)
            end
        end
        @testset "MatrixReshaped rank" begin
            for (d, s) in zip(d1s, sizes)
                @test rank(d) == minimum(s)
            end
        end
        @testset "MatrixReshaped insupport" begin
            for (i, d) in enumerate(d1s[1:end-1])
                for (j, s) in enumerate(sizes[1:end-1])
                    @test (i == j) ⊻ !insupport(d, reshape(x1, s))
                end
            end
        end
        @testset "MatrixReshaped mean" begin
            for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
                @test mean(d) == reshape(mean(d1), s)
            end
        end
        @testset "MatrixReshaped mode" begin
            for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
                @test mode(d) == reshape(mode(d1), s)
            end
        end
        @testset "MatrixReshaped covariance" begin
            for (d, (n, p)) in zip(d1s[1:end-1], sizes[1:end-1])
                @test cov(d) == cov(d1)
                @test cov(d, Val(false)) == reshape(cov(d1), n, p, n, p)
            end
        end
        @testset "MatrixReshaped variance" begin
            for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
                @test var(d) == reshape(var(d1), s)
            end
        end
        @testset "MatrixReshaped params" begin
            for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
                @test params(d) == (d1, s)
            end
        end
        @testset "MatrixReshaped partype" begin
            for d in d1s
                @test partype(d) === partype(d1)
            end
        end
        @testset "MatrixReshaped eltype" begin
            for d in d1s
                @test eltype(d) === eltype(d1)
            end
        end
        @testset "MatrixReshaped logpdf" begin
            for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
                x = reshape(x1, s)
                @test logpdf(d, x) == logpdf(d1, x1)
            end
        end
        @testset "MatrixReshaped rand" begin
            for d in d1s
                x = rand(rng, d)
                @test insupport(d, x)
                @test insupport(d1, vec(x))
                @test logpdf(d, x) == logpdf(d1, vec(x))
            end
        end
        @testset "MatrixReshaped vec" begin
            for d in d1s
                @test vec(d) === d1
            end
        end
    end
end

# Note: In contrast to `@deprecate`, `@deprecate_binding` can't be tested with `@test_deprecated`
# Ref: https://github.com/JuliaLang/julia/issues/38780
@testset "matrixreshaped.jl" begin
    @testset "MvNormal" begin
        σ = rand(rng, 16, 16)
        μ = rand(rng, 16)
        d1 = MvNormal(μ, σ * σ')
        sizes = [(4, 4), (8, 2), (2, 8), (1, 16), (16, 1), (4,)]
        _redirect_stderr(devnull) do
            test_matrixreshaped(rng, d1, sizes)
        end
    end

    # Dirichlet
    @testset "Dirichlet" begin
        α = rand(rng, 36) .+ 1 # mode is only defined if all alpha > 1
        d1 = Dirichlet(α)
        sizes = [(6, 6), (4, 9), (9, 4), (3, 12), (12, 3), (1, 36), (36, 1), (6,)]
        _redirect_stderr(devnull) do
            test_matrixreshaped(rng, d1, sizes)
        end
    end
end
