using Distributions
using JSON, ForwardDiff, Calculus, PDMats, Compat # test dependencies
using Compat.Test
using Compat.Distributed
using Compat.Random
using StatsBase

tests = [
    "types",
    "utils",
    "samplers",
    "categorical",
    "univariates",
    "continuous",
    "fit",
    "multinomial",
    "binomial",
    "poissonbinomial",
    "dirichlet",
    "dirichletmultinomial",
    "mvnormal",
    "mvlognormal",
    "mvtdist",
    "kolmogorov",
    "edgeworth",
    "matrix",
    "vonmisesfisher",
    "conversion",
    "mixture",
    "gradlogpdf",
    "truncate",
    "noncentralt",
    "locationscale",
    "quantile_newton",
    "semicircle",
    "qq",
    "truncnormal",
]

printstyled("Running tests:\n", color=:blue)

if nworkers() > 1
    rmprocs(workers())
end

if Base.JLOptions().code_coverage == 1
    addprocs(Sys.CPU_CORES, exeflags = ["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
else
    addprocs(Sys.CPU_CORES, exeflags = "--check-bounds=yes")
end

@everywhere using Compat.Random
@everywhere srand(345679)
res = pmap(tests) do t
    @eval module $(Symbol("Test_", t))
    using Distributions
    using JSON, ForwardDiff, Calculus, PDMats, Compat # test dependencies
    using Compat.Test
    using Compat.Random
    using Compat.LinearAlgebra
    using StatsBase
    include($t * ".jl")
    end
    return
end

# print method ambiguities
println("Potentially stale exports: ")
display(Compat.Test.detect_ambiguities(Distributions))
println()
