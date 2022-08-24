module TestCGF
using Test
using Distributions
import ForwardDiff

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
                   ]
        κ₀ = cgf(dist, 0)
        @test κ₀ ≈ 0 atol=2*eps(one(float(κ₀)))
        κ₁ = d(Base.Fix1(cgf, dist))(0)
        @test κ₁ ≈ mean(dist)
        κ₂ = d(d(Base.Fix1(cgf, dist)))(0)
        @test κ₂ ≈ var(dist)
        
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
