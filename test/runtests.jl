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
    "mvnormal",
    "mvlognormal",
    "mvtdist",
    "normalinversegaussian",
    "kolmogorov",
    "edgeworth",
    "matrix",
    "noncentralhypergeometric",
    "vonmisesfisher",
    "conversion",
    "mixture",
    "gradlogpdf",
    "truncate",
    "truncatednormal",
	"generalizedextremevalue"]

print_with_color(:blue, "Running tests:\n")

if nworkers() > 1
    rmprocs(workers())
end

if Base.JLOptions().code_coverage == 1
    addprocs(Sys.CPU_CORES, exeflags = ["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
else
    addprocs(Sys.CPU_CORES, exeflags = "--check-bounds=yes")
end

using Distributions
@everywhere srand(345679)
res = pmap(tests) do t
    include(t*".jl")
    nothing
end

# in v0.5, pmap returns the exception, but doesn't throw it, so we need
# to test and rethrow
if VERSION < v"0.5.0-"
    map(x -> isa(x, Exception) ? throw(x) : nothing, res)
end
