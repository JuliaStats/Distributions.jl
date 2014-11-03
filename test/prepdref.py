# A Python script to prepare reference values for distribution testing

import re
from scipy.stats import *

def parse_distr(s):
	"""Parse a string into (distr_name, params)"""

	l = s.index("(")
	r = s.index(")")
	name = s[:l]
	terms = re.split(r",\s*", s[l+1:r])
	args = tuple(float(t) for t in terms)
	return name, args


def read_distr_list(filename):
	"""Read a list of Julia distributions from a text file"""

	with open(filename) as f:
		lines = f.readlines()

	lst = []
	for line in lines:
		s = line.strip()
		name, args = parse_distr(s)
		lst.append((s, name, args))

	return lst


def to_scipy_dist(name, args):
	"""Convert from Julia distribution to scipy.stats distribution"""

	if name == "Bernoulli":
		assert len(args) == 1
		return bernoulli(args[0])

	elif name == "Binomial":
		assert len(args) == 2
		return binom(args[0], args[1])

	elif name == "DiscreteUniform":
		assert len(args) == 2
		return randint(args[0], args[1] + 1)

	elif name == "Geometric":
		assert len(args) == 1
		return geom(args[0])

	elif name == "Hypergeometric":
		assert len(args) == 3
		return hypergeom(args[0] + args[1], args[0], args[2])

	elif name == "NegativeBinomial":
		assert len(args) == 2
		return nbinom(args[0], args[1])

	elif name == "Poisson":
		assert len(args) == 1
		return poisson(args[0])

	else:
		raise ValueError("Unknown distribution name %s" % name)



def do_main(filename):
	"""main skeleton"""

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

			lp25 = d.logpmf(x25)
			lp50 = d.logpmf(x50)
			lp75 = d.logpmf(x75)

			# workaround inconsistency of definitions
			if name == "Geometric":
				x25 -= 1
				x50 -= 1
				x75 -= 1
				m -= 1

			print >>fout, '"%s", %.16e, %.16e, %.16e, %d, %d, %d, %.16e, %.16e, %.16e' % (
				ex, m, v, ent, x25, x50, x75, lp25, lp50, lp75)


if __name__ == "__main__":
	do_main("discrete_ref")

