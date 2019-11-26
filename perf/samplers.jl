# Benchmarking samplers

using BenchmarkTools
using Distributions

import Random

if haskey(ENV, "allsamplers") || haskey(ENV, "CI")
    for s in ["categorical", "binomial", "poisson", "exponential", "gamma"]
        ENV[s] = ""
    end
end

import Distributions: AliasTable, CategoricalDirectSampler

make_sampler(::Type{<:CategoricalDirectSampler}, k::Integer) = CategoricalDirectSampler(fill(1/k, k))
make_sampler(::Type{<:AliasTable}, k::Integer) = AliasTable(fill(1/k, k))

if haskey(ENV, "categorical") && ENV["categorical"] != "skip"
    @info "Categorical"
            
    for ST in [CategoricalDirectSampler, AliasTable]
        mt = Random.MersenneTwister(33)
        @info string(ST)
        ks = haskey(ENV, "CI") ? 2 .^ (1:3) : 2 .^ (1:12)
        for k in ks
            s = make_sampler(ST, k)
            b = @benchmark rand($mt, $s)
            @info "k: $k, result: $b"
        end
    end
end


if haskey(ENV, "binomial") && ENV["binomial"] != "skip"
    @info "Binomial"
    
    import Distributions: BinomialAliasSampler, BinomialGeomSampler, BinomialTPESampler, BinomialPolySampler
    
    for ST in [BinomialAliasSampler,
                   BinomialGeomSampler, 
                   BinomialTPESampler, 
                   BinomialPolySampler]
        mt = Random.MersenneTwister(33)
        @info string(ST)
        nvals = haskey(ENV, "CI") ? [2] : 2 .^ (1:12)
        pvals = haskey(ENV, "CI") ? [0.3] : [0.3, 0.5, 0.9]
        for n in nvals, p in pvals
            s = ST(n, p)
            b = @benchmark rand($mt, $s)
            @info "(n,p): $((n,p)), result: $b"
        end
    end
end

if haskey(ENV, "poisson") && ENV["poisson"] != "skip"
    @info "Poisson samplers"
    
    import Distributions: PoissonCountSampler, PoissonADSampler
    
    for ST in [PoissonCountSampler, PoissonADSampler]
        @info string(ST)
        mt = Random.MersenneTwister(33)
        µs = haskey(ENV, "CI") ? [5.0] : [5.0, 10.0, 15.0, 20.0, 30.0]
        for μ in µs
            s = ST(µ)
            b = @benchmark rand($mt, $s)
            @info "µ: $µ, result: $b"
        end
    end
end

if haskey(ENV, "exponential") && ENV["exponential"] != "skip"
    @info "Exponential"
    
    import Distributions: ExponentialSampler, ExponentialLogUSampler
    
    for ST in (ExponentialSampler, ExponentialLogUSampler)
        @info string(ST)
        mt = Random.MersenneTwister(33)
        scale_values = haskey(ENV, "CI") ? [10.0] : [10.0, 15.0, 20.0, 30.0]
        for scale in scale_values
            s = ST(scale)
            b = @benchmark rand($mt, $s)
            @info "scale: $scale, result: $b"        
        end
    end
end

if haskey(ENV, "gamma") && ENV["gamma"] != "skip"
    @info "Gamma"
    
    import Distributions: GammaGDSampler, GammaGSSampler, GammaMTSampler, GammaIPSampler
    @info "Low"
    for ST in [GammaGSSampler, GammaIPSampler]
        @info string(ST)
        mt = Random.MersenneTwister(33)
        αs = haskey(ENV, "CI") ? [0.1] : [0.1, 0.5, 0.9]
        for α in αs
            g = Gamma(α, 1.0)
            s = ST(g)
            b = @benchmark rand($mt, $s)
            @info "α: $α, result: $b"
        end
    end
    @info "High"    
    for ST in [GammaMTSampler, GammaGDSampler]
        @info string(ST)
        mt = Random.MersenneTwister(33)
        αs = haskey(ENV, "CI") ? [1.5] : [1.5, 2.0, 3.0, 5.0, 20.0]
        for α in αs
            g = Gamma(α, 1.0)
            s = ST(g)
            b = @benchmark rand($mt, $s)
            @info "α: $α, result: $b"
        end
    end
end
