# A Python script to prepare reference values for distribution testing

import re
import numpy as np
from numpy import sqrt, nan, inf, ceil
from scipy.stats import *
import json

def parse_dentry(s):
	"""Parse a string into (distr_name, args)"""

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


def read_dentry_list(filename):
	"""Read a list of Julia distribution entries from a text file"""

	with open(filename) as f:
		lines = f.readlines()

	lst = []
	for line in lines:
		s = line.strip()
		if len(s) == 0 or s.startswith("#"):
			continue
		name, args = parse_dentry(s)
		lst.append((s, name, args))

	return lst


def dsamples(a, b):
	nmax = 10
	n = b - a + 1
	if n < nmax:
		return np.arange(a, b+1)
	else:
		return np.round(np.linspace(a, b, nmax)).astype(int)


def get_dinfo(distr_name, args):
	"""Make an python object that captures all relevant quantities

		Returns a tuple in the form of (d, supp, xs, pdict), where:

		- d: the scipy.stats distribution object
		- supp: the support in the form of (minimum, maximum)
		        Note that "inf" or "-inf" should be used for infinity values
		- xs: a list of samples at which logpdf and cdf are to be tested
		- pdict: a dictionary of distribution parameters to check
	"""

	if distr_name == "Bernoulli":
		assert len(args) <= 1
		p = args[0] if len(args) == 1 else 0.5
		d = bernoulli(p)
		return (d, (0, 1), [0, 1], { 
			"succprob" : p, 
			"failprob": 1.0 - p})

	elif distr_name == "Binomial":
		assert len(args) <= 2
		n = args[0] if len(args) >= 1 else 1
		p = args[1] if len(args) >= 2 else 0.5
		d = binom(n, p)

		return (d, (0, n), dsamples(0, n), {
			"succprob" : p,
			"failprob" : 1.0 - p,
			"ntrials" : n})

	elif distr_name == "DiscreteUniform":
		assert len(args) <= 2
		if len(args) == 0:
			a, b = 0, 1
		elif len(args) == 1:
			a, b = 0, args[0]
		else:
			a, b = args
		d = randint(a, b+1)

		return (d, (a, b), dsamples(a, b), {})

	elif distr_name == "Geometric":
		assert len(args) <= 1
		p = args[0] if len(args) == 1 else 0.5
		d = geom(p)

		n = int(np.ceil(d.ppf(0.99)))
		return (d, (0, "inf"), dsamples(1, n), {})

	elif distr_name == "Hypergeometric":
		assert len(args) == 3
		ns, nf, n = args
		d = hypergeom(ns + nf, ns, n)
		a = max(n - nf, 0)
		b = min(ns, n)
		return (d, (a, b), dsamples(a, b), {})

	elif distr_name == "NegativeBinomial":
		assert len(args) <= 2
		r = args[0] if len(args) >= 1 else 1
		p = args[1] if len(args) >= 2 else 0.5
		d = nbinom(r, p)
		n = int(np.ceil(d.ppf(0.99)))
		return (d, (0, "inf"), dsamples(0, n), {})

	elif distr_name == "Poisson":
		assert len(args) <= 1
		lam = args[0] if len(args) == 1 else 1.0
		d = poisson(lam)
		n = int(np.ceil(d.ppf(0.99)))
		return (d, (0, "inf"), dsamples(0, n), {})

	else:
		raise ValueError("Unrecognized distribution name: " + distr_name)



def make_json(ex, c, distr_name, args, d, mm, xs, pdict):
	"""Make a json object by collecting all information"""

	if c == "discrete":
		is_discrete = True
	elif c == "continuous":
		is_discrete = False
	else:
		raise ValueError("Invalid value of the c-argument.")

	jdict = {"dtype" : distr_name,
			"params" : pdict,
			"minimum" : mm[0],
			"maximum" : mm[1],
			"mean" : d.mean(),
			"var" : d.var(),
			"entropy" : np.float64(d.entropy()),
			"median" : d.median(), 
			"q10" : d.ppf(0.10), 
			"q25" : d.ppf(0.25), 
			"q50" : d.ppf(0.50), 
			"q75" : d.ppf(0.75), 
			"q90" : d.ppf(0.90)} 

	if is_discrete:
		pts = [{"x" : x, "logpdf" : d.logpmf(x), "cdf" : d.cdf(x)} for x in xs]
	else:
		pts = [{"x" : x, "logpdf" : d.logpdf(x), "cdf" : d.cdf(x)} for x in xs]

	jdict["points"] = pts

	# work around inconsistencies in scipy.stats
	if distr_name == "Bernoulli":
		if len(args) == 0 or args[0] == 0.5:
			jdict["median"] = 0.5

	elif distr_name == "Geometric":
		for t in ["mean", "median", "q10", "q25", "q50", "q75", "q90"]:
			jdict[t] -= 1

		for pt in pts:
			pt["x"] -= 1

	# output
	return [ex, jdict]


def do_main(c):
	"""The main driver, c can be either 'discrete' or 'continuous'."""

	srcfile = "%s_test.lst" % c
	dstfile = "%s_test.json" % c
	entries = read_dentry_list(srcfile)

	jall = []
	for (ex, dname, args) in entries:
		print ex, "..."
		d, mm, xs, pdict = get_dinfo(dname, args)
		je = make_json(ex, c, dname, args, d, mm, xs, pdict)

		# add je to list
		jall.append(je)

	with open(dstfile, "wt") as fout:
		print >>fout, json.JSONEncoder(indent=2, sort_keys=True).encode(jall)


if __name__ == "__main__":
	do_main("discrete")
	# do_main("continuous")


