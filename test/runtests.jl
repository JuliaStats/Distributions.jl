using Distributions
using JSON, ForwardDiff, Calculus, PDMats, Compat # test dependencies
using Test

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
]

print_with_color(:blue, "Running tests:\n")

if nworkers() > 1
    rmprocs(workers())
end

if Base.JLOptions().code_coverage == 1
    addprocs(Sys.CPU_CORES, exeflags = ["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
else
    addprocs(Sys.CPU_CORES, exeflags = "--check-bounds=yes")
end

@everywhere using Distributions
@everywhere using JSON, ForwardDiff, Calculus, PDMats, Compat # test dependencies
@everywhere using Test
@everywhere srand(345679)
res = pmap(tests) do t
    include(t*".jl")
    nothing
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
