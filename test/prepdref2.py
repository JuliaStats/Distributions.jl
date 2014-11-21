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


def dsamples(d, rmin, rmax):
	vmin = rmin if np.isfinite(rmin) else int(np.floor(d.ppf(0.01)))
	vmax = rmax if np.isfinite(rmax) else int(np.ceil(d.ppf(0.99)))

	if vmax - vmin + 1 <= 10:
		xs = range(vmin, vmax+1)
	else:
		xs = [int(np.round(d.ppf(q))) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
		xs = list(np.unique(xs))
		if vmin < xs[0]:
			xs.insert(0, vmin)
		if vmax > xs[-1]:
			xs.append(vmax)

	return xs


def csamples(d):
	return [d.ppf(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]


def get(a, i):
	"""Retrieve the i-th element or return None"""

	return a[i] if len(a) > i else None


def get_dinfo(dname, args):
	"""Make an python object that captures all relevant quantities

		Returns a tuple in the form of (d, supp, xs, pdict), where:

		- d: the scipy.stats distribution object
		- supp: the support in the form of (minimum, maximum)
		        Note that "inf" or "-inf" should be used for infinity values
		- pdict: a dictionary of distribution parameters to check
	"""

	if dname == "Arcsine":
		assert len(args) == 0
		return (arcsine(), (0.0, 1.0), {})

	elif dname == "Beta":
		assert len(args) <= 2
		if len(args) <= 1:
			a = b = get(args, 0) or 1.0
		else:
			a, b = args
		return (beta(a, b), (0.0, 1.0), {})

	elif dname == "BetaPrime":
		assert len(args) <= 2
		if len(args) <= 1:
			a = b = get(a, 0) or 1.0
		else:
			a, b = args
		return (betaprime(a, b), (0.0, inf), {})

	elif dname == "Bernoulli":
		assert len(args) <= 1
		p = get(args, 0) or 0.5
		return (bernoulli(p), (0, 1), {"succprob" : p, "failprob": 1.0 - p})

	elif dname == "Binomial":
		assert len(args) <= 2
		n = int(get(args, 0) or 1)
		p = get(args, 1) or 0.5
		return (binom(n, p), (0, n), 
			{"succprob" : p, "failprob" : 1.0 - p, "ntrials" : n})

	elif dname == "Cauchy":
		assert len(args) <= 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (cauchy(l, s), (-inf, inf), {})

	elif dname == "Chi":
		assert len(args) == 1
		df = args[0]
		return (chi(df), (0, inf), {})

	elif dname == "Chisq":
		assert len(args) == 1
		df = args[0]
		return (chi2(df), (0, inf), {})

	elif dname == "DiscreteUniform":
		assert len(args) <= 2
		if len(args) == 0:
			a, b = 0, 1
		elif len(args) == 1:
			a, b = 0, int(args[0])
		else:
			a, b = int(args[0]), int(args[1])
		return (randint(a, b+1), (a, b), {})

	elif dname == "Erlang":
		assert len(args) <= 2
		a = get(args, 0) or 1
		s = get(args, 1) or 1.0
		return (erlang(a, scale=s), (0, inf), {})

	elif dname == "Exponential":
		assert len(args) <= 1
		s = get(args, 0) or 1.0
		return (expon(scale=s), (0, inf), {})

	elif dname == "Gamma":
		assert len(args) <= 2
		a = get(args, 0) or 1.0
		s = get(args, 1) or 1.0
		return (gamma(a, scale=s), (0, inf), {})

	elif dname == "Geometric":
		assert len(args) <= 1
		p = get(args, 0) or 0.5
		return (geom(p), (0, inf), {})

	elif dname == "Gumbel":
		assert len(args) <= 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (gumbel_r(l, s), (-inf, inf), {})

	elif dname == "Hypergeometric":
		assert len(args) == 3
		ns, nf, n = [int(t) for t in args]
		return (hypergeom(ns + nf, ns, n), 
			(max(n - nf, 0), min(ns, n)), {})

	elif dname == "InverseGamma":
		assert len(args) <= 2
		a = get(args, 0) or 1.0
		s = get(args, 1) or 1.0
		return (invgamma(a, scale=s), (0, inf), {})

	elif dname == "Laplace":
		assert len(args) <= 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (laplace(l, s), (-inf, inf), {})

	elif dname == "Logistic":
		assert len(args) == 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (logistic(l, s), (-inf, inf), {})

	elif dname == "NegativeBinomial":
		assert len(args) <= 2
		r = int(get(args, 0) or 1)
		p = get(args, 1) or 0.5
		return (nbinom(r, p), (0, inf), {})

	elif dname == "Poisson":
		assert len(args) <= 1
		lam = get(args, 0) or 1.0
		return (poisson(lam), (0, inf), {})

	else:
		raise ValueError("Unrecognized distribution name: " + dname)


def json_num(x):
	return x if np.isfinite(x) else str(x)


def make_json(ex, c, distr_name, args, d, mm, pdict):
	"""Make a json object by collecting all information"""

	if c == "discrete":
		is_discrete = True
	elif c == "continuous":
		is_discrete = False
	else:
		raise ValueError("Invalid value of the c-argument.")

	r_min, r_max = mm
	jdict = {"dtype" : distr_name,
			"params" : pdict,
			"minimum" : json_num(r_min),
			"maximum" : json_num(r_max),
			"mean" : json_num(d.mean()),
			"var" : json_num(d.var()),
			"entropy" : np.float64(d.entropy()),
			"median" : d.median(), 
			"q10" : d.ppf(0.10), 
			"q25" : d.ppf(0.25), 
			"q50" : d.ppf(0.50), 
			"q75" : d.ppf(0.75), 
			"q90" : d.ppf(0.90)} 

	if is_discrete:
		if distr_name == "Geometric":
			xs = dsamples(d, 1, inf)
		else:
			xs = dsamples(d, r_min, r_max)
		pts = [{"x" : x, "logpdf" : d.logpmf(x), "cdf" : d.cdf(x)} for x in xs]
	else:
		xs = csamples(d)
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

	elif distr_name == "Cauchy":
		jdict["mean"] = "nan"
		jdict["var"] = "nan"

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
		d, mm, pdict = get_dinfo(dname, args)
		je = make_json(ex, c, dname, args, d, mm, pdict)

		# add je to list
		jall.append(je)

	with open(dstfile, "wt") as fout:
		print >>fout, json.JSONEncoder(indent=2, sort_keys=True).encode(jall)


if __name__ == "__main__":
	do_main("discrete")
	do_main("continuous")


