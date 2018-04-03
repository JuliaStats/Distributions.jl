using Distributions
using JSON, ForwardDiff, Calculus, PDMats, Compat # test dependencies
using Compat.Test


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

@everywhere srand(345679)
res = pmap(tests) do t
    @eval module $(Symbol("Test_", t))
    using Distributions
    using JSON, ForwardDiff, Calculus, PDMats, Compat # test dependencies
    using Base.Test
    include($t * ".jl")
    end
    return
end

# print method ambiguities
println("Potentially stale exports: ")
display(Compat.Test.detect_ambiguities(Distributions))
println()
