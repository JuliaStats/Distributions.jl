# Benchmarking samplers

using BenchmarkLite
using Distributions

### define benchmark tasks

abstract type UnivariateSamplerRun{Spl} <: Proc end
mutable struct BatchSamplerRun{Spl} <: UnivariateSamplerRun{Spl} end
mutable struct IndivSamplerRun{Spl} <: UnivariateSamplerRun{Spl} end

const batch_unit = 1000

Base.isvalid(::UnivariateSamplerRun, cfg) = true
Base.length(p::UnivariateSamplerRun, cfg) = batch_unit
Base.string(p::UnivariateSamplerRun{Spl}) where {Spl} = getname(Spl)

Base.start(p::BatchSamplerRun{Spl}, cfg) where {Spl} = getsampler(Spl, cfg)
Base.start(p::IndivSamplerRun{Spl}, cfg) where {Spl} = ()
Base.done(p::UnivariateSamplerRun, cfg, s) = nothing

getsampler(::Type{Spl}, cfg) where {Spl<:Sampleable} = Spl(cfg...)

function Base.run(p::BatchSamplerRun, cfg, s) 
    for i = 1:batch_unit
        rand(s)
    end
end

function Base.run(p::IndivSamplerRun{Spl}, cfg, s) where Spl 
    for i = 1:batch_unit
        rand(getsampler(Spl,cfg))
    end
end


make_procs(P,spltypes) = Proc[P{T}() for T in spltypes]



### specific benchmarking program

## categorical

import Distributions: AliasTable, CategoricalDirectSampler
getsampler(::Type{CategoricalDirectSampler}, k::Int) = CategoricalDirectSampler(fill(1/k, k))
getsampler(::Type{AliasTable}, k::Int) = AliasTable(fill(1/k, k))

getname(::Type{CategoricalDirectSampler}) = "direct"
getname(::Type{AliasTable}) = "alias"

benchmark_categorical(P) = (
    (CategoricalDirectSampler, AliasTable),
    "K", 2 .^ (1:12))

## binomial

import Distributions: BinomialRmathSampler, BinomialAliasSampler
import Distributions: BinomialGeomSampler, BinomialTPESampler, BinomialPolySampler

getname(::Type{BinomialRmathSampler}) = "rmath"
getname(::Type{BinomialAliasSampler}) = "alias"
getname(::Type{BinomialGeomSampler}) = "Geom"
getname(::Type{BinomialTPESampler}) = "BTPE"
getname(::Type{BinomialPolySampler}) = "Poly"

benchmark_binomial(P) = (
    (BinomialRmathSampler,
               BinomialAliasSampler,
               BinomialGeomSampler, 
               BinomialTPESampler, 
               BinomialPolySampler),
    "(n, p)", vec([(n, p) for p in [0.3, 0.5, 0.9], n in 2.^(1:12)]))

## poisson

import Distributions: PoissonRmathSampler, PoissonCountSampler, PoissonADSampler

getname(::Type{PoissonRmathSampler}) = "rmath"
getname(::Type{PoissonCountSampler}) = "count"
getname(::Type{PoissonADSampler}) = "AD"

Base.isvalid(::BatchSamplerRun{PoissonADSampler}, mu) = (mu >= 5.0)

benchmark_poisson() = (
    (PoissonRmathSampler, PoissonCountSampler, PoissonADSampler),
    "μ", [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0])

## exponential

import Distributions: ExponentialSampler, ExponentialLogUSampler

getname(::Type{ExponentialSampler}) = "base"
getname(::Type{ExponentialLogUSampler}) = "logu"

benchmark_exponential() = (
    (ExponentialSampler, ExponentialLogUSampler),
    "scale", [1.0])

## gamma

import Distributions: GammaRmathSampler, GammaGDSampler, GammaGSSampler,
    GammaMTSampler, GammaIPSampler

getname(::Type{GammaRmathSampler}) = "rmath"
getname(::Type{GammaGDSampler}) = "GD"
getname(::Type{GammaGSSampler}) = "GS"
getname(::Type{GammaMTSampler}) = "MT"
getname(::Type{GammaIPSampler}) = "IP"

benchmark_gamma_hi() = (
    (GammaRmathSampler, GammaMTSampler, GammaGDSampler),
    "Dist", [(Gamma(α, 1.0),) for α in [1.5, 2.0, 3.0, 5.0, 20.0]])

benchmark_gamma_lo() = (
    (GammaRmathSampler, GammaGSSampler, GammaIPSampler),
    "Dist", [(Gamma(α, 1.0),) for α in [0.1, 0.5, 0.9]])

### main

const dnames = ["categorical", 
                "binomial", 
                "poisson",
                "exponential",
                "gamma_hi","gamma_lo"]

function printhelp()
    println("Require exactly two arguments. Usage:")
    println()
    println("   julia <path>/sampler.jl <proctype> <distrname>")
    println()
    println("   <proctype> can be:")
    println()
    println("      - batch : tests iteration of the sampler")
    println("      - indiv : tests construction and iteration of the sampler")
    println()
    println("   <distrname> can be:")
    println()
    for dname in dnames
        println("      - $dname")
    end
    println()
end

function getarg(args)
    if length(args) == 2
        ptype = args[1]
        if !(ptype in ["batch","indiv"])
            printhelp()
            exit(1)
        end        
        dname = args[2]
        if !(dname in dnames)
            printhelp()
            exit(1)
        end
        return ptype, dname
    else
        printhelp()
        exit(1)
    end
end

ptype, dname = getarg(ARGS)

function do_benchmark(ptype, dname; verbose::Int=2)
    if ptype == "batch"
        proc = BatchSamplerRun
    else
        proc = IndivSamplerRun
    end
    (spltypes, cfghead, cfgs) = 
        dname == "categorical" ? benchmark_categorical() :
        dname == "binomial"    ? benchmark_binomial() :
        dname == "poisson"     ? benchmark_poisson() :
        dname == "exponential" ? benchmark_exponential() :
        dname == "gamma_hi"    ? benchmark_gamma_hi() :
        dname == "gamma_lo"    ? benchmark_gamma_lo() :
        error("benchmarking function for $dname has not been implemented.")

    r = run(make_procs(proc,spltypes), cfgs; duration=0.5, verbose=verbose)
    println()
    show(r; unit=:mps, cfghead=cfghead)
end

do_benchmark(ptype, dname)
println()

