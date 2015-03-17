tests = [
    "types",
    "utils",
    "samplers",
    "categorical",
    "univariates", 
    "fit", 
    "multinomial",
    "poissonbinomial",
    "dirichlet",
    "mvnormal",
    "mvtdist",
    "kolmogorov",
    "edgeworth",
    "matrix",
    "noncentralhypergeometric",
    "vonmisesfisher",
    "compoundvariate",
    "conjugates",
    "conjugates_normal",
    "conjugates_mvnormal",
    "mixture",
    "gradlogpdf",
    "tvpack"]

print_with_color(:blue, "Running tests:\n")

srand(345678)

for t in tests
    test_fn = "$t.jl"
    print_with_color(:green, "* $test_fn\n")
    include(test_fn)
end
