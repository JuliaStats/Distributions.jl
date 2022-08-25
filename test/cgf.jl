module TestCGF
using Test
using Distributions
import ForwardDiff

@testset "cgf uniform around 0" begin
    for (lo, hi, t) in [
        ((Float16(0), Float16(1), sqrt(eps(Float16)))),
        ((Float16(0), Float16(1), Float16(0))),
        ((Float16(0), Float16(1), -sqrt(eps(Float16)))),
        (0f0, 1f0, sqrt(eps(Float32))),
        (0f0, 1f0, 0f0),
        (0f0, 1f0, -sqrt(eps(Float32))),
        (-2f0, 1f0, 1f-30),
        (-2f-4, -1f-4, -2f-40),
        (0.0, 1.0, sqrt(eps(Float64))),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, -sqrt(eps(Float64))),
        (-2.0, 5.0, -1e-35),
                       ]
        T = typeof(lo)
        @assert T == typeof(lo) == typeof(hi) == typeof(t)
        @assert t <= sqrt(eps(T))
        d = Uniform(lo, hi)
        precision = 512
        d_big = Uniform(BigFloat(lo, precision=precision), BigFloat(hi; precision=precision))
        t_big = BigFloat(t, precision=precision)
        @test cgf(d, t) isa T
        if iszero(t)
            @test cgf(d,t) === zero(t)
        else
            @test Distributions.cgf_around_zero(d, t) ≈ Distributions.cgf_away_from_zero(d_big, t_big) atol=eps(t) rtol=0
            @test Distributions.cgf_around_zero(d, t) === cgf(d, t)
        end
    end
end

@testset "CumulantGeneratingFunctions.jl" begin
    d(f) = Base.Fix1(ForwardDiff.derivative, f)
    @testset "$(dist)" for (dist, ts) in [
        (Dirac(13),                Any[1, 1f-4, 1e10, 10, -4]),
        (Dirac(-1f2),              Any[1, 1f-4, 1e10, 10,-4]),
        (Bernoulli(0.5),           Any[1f0, -1f0,1e6, -1e6]),
        (Bernoulli(0.1),           Any[1f0, -1f0,1e6, -1e6]),
        (Binomial(10,0.1),         Any[1f0, -1f0,1e6, -1e6]),
        (Binomial(100,1e-3),       Any[1f0, -1f0,1e6, -1e6]),
        (Geometric(0.1),           Any[1f-1, -1e6]           ),
        (Geometric(0.5),           Any[1f-1, -1e6]           ),
        (Geometric(0.5),           Any[1f-1, -1e6]           ),
        (NegativeBinomial(10,0.5), Any[-1f0, -200.0,-1e6]),
        (NegativeBinomial(3,0.1),  Any[-1f0, -200.0,-1e6] ),
        (Poisson(1),               Any[1f0,2f0,10.0,50.0]),
        (Poisson(10),              Any[1f0,2f0,10.0,50.0]),
        (Poisson(1e-3),            Any[1f0,2f0,10.0,50.0]),
        (Uniform(0,1),             Any[1, -1, 100f0, 1e6, -1e6]),
        (Uniform(100f0,101f0),     Any[1, -1, 100f0, 1e6, -1e6]),
        (Normal(0,1),              Any[1, -1, 100f0, 1e6, -1e6]),
        (Normal(1,0.4),            Any[1, -1, 100f0, 1e6, -1e6]),
        (Exponential(1),           Any[0.9, -1, -100f0, -1e6]),
        (Exponential(0.91),        Any[0.9, -1, -100f0, -1e6]),
        (Exponential(10),          Any[0.08, -1, -100f0, -1e6]),
        (Gamma(1,1),               Any[0.9, -1, -100f0, -1e6]),
        (Gamma(10,1),              Any[0.9, -1, -100f0, -1e6]),
        (Gamma(0.2, 10),           Any[0.08, -1, -100f0, -1e6]),
        (Laplace(1, 1),            Any[0.99, -0.99, 1f-2, -1f-5]),
        (Chisq(1),                 Any[0.49, -1, -100, -1f6]),
        (Chisq(3),                 Any[0.49, -1, -100, -1f6]),
        (NoncentralChisq(3,2),     Any[0.49, -1, -100, -1f6]),
        (Logistic(0,1),            Any[-0.99,0.99, 1f-2, -1f-2]),
        (Logistic(100,10),         Any[-0.099,0.099, 1f-2, -1f-2]),
        (Erlang(1,0.4),            Any[1, 1/0.400001, -1, -100f0, -1e6]),
        (Erlang(10,0.01),            Any[1, 1/0.010001f0, -1, -100f0, -1e6]),
                   ]
        κ₀ = cgf(dist, 0)
        @test κ₀ ≈ 0 atol=2*eps(one(float(κ₀)))
        κ₁ = d(Base.Fix1(cgf, dist))(0)
        @test κ₁ ≈ mean(dist)
        if VERSION >= v"1.4"
            κ₂ = d(d(Base.Fix1(cgf, dist)))(0)
            @test κ₂ ≈ var(dist)
        end
        for t in ts
            val = @inferred cgf(dist, t)
            @test isfinite(val)
            if isfinite(mgf(dist, t))
                rtol = eps(float(one(t)))^(1/2)
                @test (exp∘cgf)(dist, t) ≈ mgf(dist, t) rtol=rtol
            end
        end
    end
end

end#module
