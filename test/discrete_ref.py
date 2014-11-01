# Generate reference values based on scipy.stats

from scipy.stats import *

lst = [
	(bernoulli(0.5), "Bernoulli(0.5)"), 
	(bernoulli(0.9), "Bernoulli(0.9)"),
	(bernoulli(0.1), "Bernoulli(0.1)"),
	(binom(1, 0.5), "Binomial(1, 0.5)"),
	(binom(100, 0.1), "Binomial(100, 0.1)"),
	(binom(100, 0.9), "Binomial(100, 0.9)"),
	(randint(0, 4+1), "DiscreteUniform(0, 4)"),
	(randint(2, 8+1), "DiscreteUniform(2, 8)"),
	(geom(0.1), "Geometric(0.1)"),
	(geom(0.5), "Geometric(0.5)"),
	(geom(0.9), "Geometric(0.9)"),
	(hypergeom(2+2, 2, 2), "Hypergeometric(2, 2, 2)"),
	(hypergeom(3+2, 3, 2), "Hypergeometric(3, 2, 2)"),
	(hypergeom(4+5, 4, 6), "Hypergeometric(4, 5, 6)"),
	(hypergeom(60+80, 60, 100), "Hypergeometric(60, 80, 100)"),
	(nbinom(1, 0.5), "NegativeBinomial(1, 0.5)"),
	(nbinom(5, 0.6), "NegativeBinomial(5, 0.6)"),
	(poisson(0.5), "Poisson(0.5)"),
	(poisson(2.0), "Poisson(2.0)"),
	(poisson(10.0), "Poisson(10.0)"),
	(poisson(80.0), "Poisson(80.0)")
]

print "distr, mean, var, entropy, x25, x50, x75, p25, p50, p75"

for (d, e) in lst:
	m = d.mean()
	v = d.var()
	ent = d.entropy()

	x25 = d.ppf(0.25)
	x50 = d.ppf(0.50)
	x75 = d.ppf(0.75)

	lp25 = d.logpmf(x25)
	lp50 = d.logpmf(x50)
	lp75 = d.logpmf(x75)

	# workaround inconsistency of definitions
	if e.startswith("Geometric("):
		x25 -= 1
		x50 -= 1
		x75 -= 1
		m -= 1

	print '"%s", %.16e, %.16e, %.16e, %d, %d, %d, %.16e, %.16e, %.16e' % (e, m, v, ent, x25, x50, x75, lp25, lp50, lp75)

