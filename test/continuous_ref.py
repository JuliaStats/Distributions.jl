# Generate reference values based on scipy.stats

from scipy.stats import *

lst = [
	(arcsine(), "Arcsine()"),
	(beta(2.0, 2.0), "Beta(2.0, 2.0)"),
	(beta(3.0, 4.0), "Beta(3.0, 4.0)"),
	(beta(17.0, 13.0), "Beta(17.0, 13.0)"),
	(betaprime(3.0, 3.0), "BetaPrime(3.0, 3.0)"),
	(betaprime(3.0, 5.0), "BetaPrime(3.0, 5.0)"),
	(betaprime(5.0, 3.0), "BetaPrime(5.0, 3.0)"),
	(cauchy(0.0, 1.0), "Cauchy(0.0, 1.0)"),
	(cauchy(10.0, 1.0), "Cauchy(10.0, 1.0)"),
	(cauchy(2.0, 10.0), "Cauchy(2.0, 10.0)"),
	(chi(1.0), "Chi(1)"),
	(chi(2.0), "Chi(2)"),
	(chi(3.0), "Chi(3)"),
	(chi(12.0), "Chi(12)"),
	(chi2(1.0), "Chisq(1)"),
	(chi2(8.0), "Chisq(8)"),
	(chi2(20.0), "Chisq(20)"),
	(erlang(1, scale=1.0), "Erlang(1, 1.0)"),
	(erlang(3, scale=1.0), "Erlang(3, 1.0)"),
	(erlang(5, scale=2.0), "Erlang(5, 2.0)"),
	(expon(scale=1.0), "Exponential(1.0)"),
	(expon(scale=6.5), "Exponential(6.5)")
]

print "distr, mean, var, entropy, x25, x50, x75, p25, p50, p75"

for (d, e) in lst:
	m = d.mean()
	v = d.var()
	ent = d.entropy()

	x25 = d.ppf(0.25)
	x50 = d.ppf(0.50)
	x75 = d.ppf(0.75)

	lp25 = d.logpdf(x25)
	lp50 = d.logpdf(x50)
	lp75 = d.logpdf(x75)

	# workaround inconsistency of definitions
	if e.startswith("Geometric("):
		x25 -= 1
		x50 -= 1
		x75 -= 1
		m -= 1

	print '"%s", %.16e, %.16e, %.16e, %.16e, %.16e, %.16e, %.16e, %.16e, %.16e' % (e, m, v, ent, x25, x50, x75, lp25, lp50, lp75)

