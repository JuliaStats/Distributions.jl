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
    "shifted"]

print_with_color(:blue, "Running tests:\n")

srand(345678)

for t in tests
    test_fn = "$t.jl"
    print_with_color(:green, "* $test_fn\n")
    include(test_fn)
end
