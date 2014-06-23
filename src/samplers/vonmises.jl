
immutable VonMisesSampler <: Sampleable{Univariate,Continuous}
    μ::Float64
    κ::Float64
    r::Float64
end

# algorithm from
#     DJ Best & NI Fisher (1979). Efficient Simulation of the von Mises
#     Distribution. Journal of the Royal Statistical Society. Series C
#     (Applied Statistics), 28(2), 152-157.
function rand(s::VonMisesSampler)
    f = 0.0
    while true
        t, u = 0.0, 0.0
        while true
            const v, w = rand() - 0.5, rand() - 0.5
            const d, e = v ^ 2, w ^ 2
            if d + e <= 0.25
                t = d / e
                u = 4 * (d + e)
                break
            end
        end
        const z = (1.0 - t) / (1.0 + t)
        f = (1.0 + s.r * z) / (s.r + z)
        const c = s.κ * (s.r - f)
        if c * (2.0 - c) > u || log(c / u) + 1 >= c
            break
        end
    end
    mod(s.μ + (rand() > 0.5 ? acos(f) : -acos(f)), twoπ)
end

immutable VonMisesNormalApproxSampler <: Sampleable{Univariate,Continuous}
    μ::Float64
    σ::Float64
end

rand(s::VonMisesNormalApproxSampler) = mod(s.μ + s.σ * randn(), twoπ)

# normal approximation for large concentrations
VonMisesSampler(μ::Float64, κ::Float64) = 
    κ > 700.0 ? VonMisesNormalApproxSampler(μ, sqrt(1.0 / κ)) :
                  begin
                      τ = 1.0 + sqrt(1.0 + 4 * κ ^ 2)
                      ρ = (τ - sqrt(2.0 * τ)) / (2.0 * κ)
                      VonMisesSampler(μ, κ, (1.0 + ρ ^ 2) / (2.0 * ρ))
                  end

