# Generate reference values based on scipy.stats

from scipy.stats import *

lst = [
	(arcsine(), "Arcsine()")
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

