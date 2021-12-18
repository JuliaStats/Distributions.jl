using Distributions
using ArraysOfArrays
using StatsBase
using LinearAlgebra
using Random
using Test


@testset "MvDiscreteNonParametric" begin

    @testset "Declaring MvDiscreteNonParametric" begin

        Random.seed!(7)
        n = 4
        m = 2
        A = nestedview(rand(n, m)')
        p = normalize!(rand(n), 1)

        # Passing probabilities
        μ = @inferred(MvDiscreteNonParametric(A, p))
        @test support(μ) == A
        @test length(μ) == m
        @test size(μ) == (m, n)
        @test probs(μ)   == p

        # Without passing probabilities
        μ = @inferred(MvDiscreteNonParametric(A))
        @test support(μ) == A
        @test length(μ) == m
        @test size(μ) == (m, n)
        @test probs(μ)   == fill(1 / n, n)

        # Array of arrays without ArraysOfArrays.jl
        n, m = 3, 2
        p = ([3 / 5, 1 / 5, 1 / 5])
        A = [[1,0],[1,1],[0,1]]
        μ = @inferred(MvDiscreteNonParametric(A, p))

        @test support(μ) == A
        @test length(μ) == m
        @test size(μ) == (length(A[1]), length(A))
        @test probs(μ) == p

    end


    # @testset "Functionalities" begin
    #     
    #     function variance(d)
    #         v = zeros(length(d))
    #         for i in 1:length(d)
    #             s = flatview(d.support)'[:,i]
    #             mₛ = mean(d)[i]
    #             v[i] = sum(abs2.(s .- mₛ), Weights(d.p))
    #         end
    #         return v
    #     end
    #     
    #     function covariance(d)
    #         n = length(d)
    #         v = zeros(n, n)
    #         for i in 1:n, j in 1:n
    #             s = flatview(d.support)'[:,i]
    #             mₛ = mean(d)[i]
    #             
    #             u = flatview(d.support)'[:,j]
    #             mᵤ = mean(d)[j]
    #             
    #             v[i,j] = sum((s .- mₛ) .* (u .- mᵤ), Weights(d.p))
    #         end
    #         return v
    #     end
    #     
    #     Random.seed!(7)
    #     n, m = 7, 9

    #     A = collect(eachrow(rand(n, m)))
    #     p = normalize!(rand(n), 1)
    #     μ = @inferred(MvDiscreteNonParametric(A, p))

    #     @test mean(μ) == mean(flatview(μ.support)[:,:], Weights(p), dims=2)[:]
    #     @test var(μ) ≈ variance(μ)
    #     @test cov(μ) ≈ covariance(μ)
    #     @test pdf(μ, flatview(μ.support)) ≈ μ.p
    #     @test pdf(μ, zeros(m)) == 0.0
    #     @test entropy(μ) == entropy(μ.p)
    #     @test entropy(μ, 2) == entropy(μ.p, 2)

    # end
    
    @testset "Sampling" begin
        Random.seed!(7)
        A = collect(eachrow(rand(10,10)))
        μ = MvDiscreteNonParametric(A)

        # for sample in eachcol(rand(μ, 30))
        #     @test sample in μ.support
        # end

        A = collect(eachrow(rand(3, 2)))
        μ = MvDiscreteNonParametric(A, [0.2,0.5,0.3])

        for i in 1:3
            samples = nestedview(rand(μ, 10000))
            @test abs(mean([A[i,:] == s for s in samples]) - μ.p[i]) < 0.05
        end
    end
        
end
        
