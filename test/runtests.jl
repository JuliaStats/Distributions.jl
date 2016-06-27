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
pmap(tests) do t
    include(t*".jl")
    nothing
end
