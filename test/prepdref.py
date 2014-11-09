# A Python script to prepare reference values for distribution testing

import re
from numpy import sqrt, nan, inf
from scipy.stats import *

def parse_distr(s):
	"""Parse a string into (distr_name, params)"""

	l = s.index("(")
	r = s.index(")")
	name = s[:l]
	ts = s[l+1:r].strip()
	if len(ts) > 0:
		terms = re.split(r",\s*", s[l+1:r])
		args = tuple(float(t) for t in terms)
	else:
		args = ()
	return name, args


def read_distr_list(filename):
	"""Read a list of Julia distributions from a text file"""

	with open(filename) as f:
		lines = f.readlines()

	lst = []
	for line in lines:
		s = line.strip()
		if s.startswith("#"):
			continue
		name, args = parse_distr(s)
		lst.append((s, name, args))

	return lst


def to_scipy_dist(name, args):
	"""Convert from Julia distribution to scipy.stats distribution"""

	if name == "Arcsine":
		assert len(args) == 0
		return arcsine()

	elif name == "Bernoulli":
		assert len(args) == 1
		return bernoulli(args[0])

	elif name == "Beta":
		assert len(args) == 2
		return beta(args[0], args[1])

	elif name == "BetaPrime":
		assert len(args) == 2
		return betaprime(args[0], args[1])

	elif name == "Binomial":
		assert len(args) == 2
		return binom(args[0], args[1])

	elif name == "Cauchy":
		assert len(args) == 2
		return cauchy(args[0], args[1])

	elif name == "Chi":
		assert len(args) == 1
		return chi(args[0])

	elif name == "Chisq":
		assert len(args) == 1
		return chi2(args[0])

	elif name == "DiscreteUniform":
		assert len(args) == 2
		return randint(args[0], args[1] + 1)

	elif name == "Erlang":
		assert len(args) == 2
		return erlang(args[0], scale=args[1])

	elif name == "Exponential":
		assert len(args) == 1
		return expon(scale=args[0])

	elif name == "Gamma":
		assert len(args) == 2
		return gamma(args[0], scale=args[1])

	elif name == "Geometric":
		assert len(args) == 1
		return geom(args[0])

	elif name == "Gumbel":
		assert len(args) == 2
		return gumbel_r(args[0], args[1])

	elif name == "Hypergeometric":
		assert len(args) == 3
		return hypergeom(args[0] + args[1], args[0], args[2])

	elif name == "InverseGamma":
		assert len(args) == 2
		return invgamma(args[0], scale=args[1])

	elif name == "Laplace":
		assert len(args) == 2
		return laplace(args[0], args[1])

	elif name == "Logistic":
		assert len(args) == 2
		return logistic(args[0], args[1])

	elif name == "NegativeBinomial":
		assert len(args) == 2
		return nbinom(args[0], args[1])

	elif name == "Normal":
		assert len(args) == 2
		return norm(args[0], args[1])

	elif name == "NormalCanon":
		assert len(args) == 2
		return norm(args[0] / args[1], sqrt(1.0 / args[1]))

	elif name == "Pareto":
		assert len(args) == 2
		return pareto(args[0], scale=args[1])

	elif name == "Poisson":
		assert len(args) == 1
		return poisson(args[0])

	elif name == "Rayleigh":
		assert len(args) == 1
		return rayleigh(scale=args[0])

	elif name == "SymTriangularDist":
		assert len(args) == 2
		return triang(0.5, loc=args[0] - args[1], scale=args[1] * 2.0)

	elif name == "TDist":
		assert len(args) == 1
		return t(args[0])

	elif name == "TruncatedNormal":
		assert len(args) == 4
		mu, sig, a, b = args
		za = (a - mu) / sig
		zb = (b - mu) / sig
		za = max(za, -1000.0)
		zb = min(zb, 1000.0)
		return truncnorm(za, zb, loc=mu, scale=sig)

	elif name == "TriangularDist":
		assert len(args) == 3
		a, b, c = args
		return triang((c - a) / (b - a), loc=a, scale=b-a)

	elif name == "Uniform":
		assert len(args) == 2
		return uniform(args[0], args[1] - args[0])

	elif name == "Weibull":
		assert len(args) == 2
		return weibull_min(args[0], scale=args[1])

	else:
		raise ValueError("Unknown distribution name %s" % name)



def do_main(title):
	"""main skeleton"""

	filename = title + "_ref"
	lst = read_distr_list(filename + ".txt")

	with open(filename + ".csv", "wt") as fout:

		for (ex, name, args) in lst:
			d = to_scipy_dist(name, args)

			m = d.mean()
			v = d.var()
			ent = d.entropy()

			x25 = d.ppf(0.25)
			x50 = d.ppf(0.50)
			x75 = d.ppf(0.75)

			if title == "discrete":
				lp25 = d.logpmf(x25)
				lp50 = d.logpmf(x50)
				lp75 = d.logpmf(x75)
			else:
				lp25 = d.logpdf(x25)
				lp50 = d.logpdf(x50)
				lp75 = d.logpdf(x75)

			# workaround inconsistency of definitions
			if name == "Geometric":
				x25 -= 1
				x50 -= 1
				x75 -= 1
				m -= 1

			if title == "discrete":
				print >>fout, '"%s", %.16e, %.16e, %.16e, %d, %d, %d, %.16e, %.16e, %.16e' % (
					ex, m, v, ent, x25, x50, x75, lp25, lp50, lp75)
			else:
				print >>fout, '"%s", %.16e, %.16e, %.16e, %.16e, %.16e, %.16e, %.16e, %.16e, %.16e' % (
					ex, m, v, ent, x25, x50, x75, lp25, lp50, lp75)


if __name__ == "__main__":
	do_main("discrete")
	do_main("continuous")


