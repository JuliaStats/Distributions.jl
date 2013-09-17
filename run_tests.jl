
tests = [
    "types",
	"utils", 
	"sample",
	"fit", 
	"discrete",
	"univariate", 
	"truncate", 
	"multinomial",
	"dirichlet", 
	"mvnormal",
	"conjugates",
	"kolmogorov",
	"edgeworth",
	"matrix",
	"vonmisesfisher",
    "conjugate-normal/normalgamma",
    "conjugate-normal/normalinversegamma",
    "conjugate-normal/normalwishart",
    "conjugate-normal/normalinversewishart",
    "conjugate-normal/normalknowncov"]

println("Running tests:")

for t in tests
	test_fn = joinpath("test", "$t.jl")
    println(" * $test_fn")
    include(test_fn)
end
