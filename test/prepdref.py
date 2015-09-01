# A Python script to prepare reference values for distribution testing

import re
import numpy as np
from numpy import sqrt, nan, inf, ceil, pi
import scipy.stats
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

		Returns a tuple in the form of (d, supp, pdict), where:

		- d: the scipy.stats distribution object
		- supp: the support in the form of (minimum, maximum)
		        Note that "inf" or "-inf" should be used for infinity values
		- pdict: a dictionary of distribution parameters to check
	"""

	if dname == "Arcsine":
		assert len(args) <= 2
		if len(args) <= 1:
			a = 0.0
			b = get(args, 0) or 1.0
		else:
			a, b = args
		return (arcsine(loc=a, scale=b-a), (a, b), {})

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
		return (cauchy(l, s), (-inf, inf), 
			{"location" : l, "scale" : s})

	elif dname == "Chi":
		assert len(args) == 1
		df = args[0]
		return (chi(df), (0, inf), {"dof" : df})

	elif dname == "Chisq":
		assert len(args) == 1
		df = args[0]
		return (chi2(df), (0, inf), {"dof" : df})

	elif dname == "Cosine":
		assert len(args) <= 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (cosine(loc=l, scale=s/pi), (l-s, l+s), 
			{"location":l, "scale":s})

	elif dname == "DiscreteUniform":
		assert len(args) <= 2
		if len(args) == 0:
			a, b = 0, 1
		elif len(args) == 1:
			a, b = 0, int(args[0])
		else:
			a, b = int(args[0]), int(args[1])
		sp = b - a + 1
		return (randint(a, b+1), (a, b), 
			{"span" : sp, "probval" : 1.0 / sp})

	elif dname == "Erlang":
		assert len(args) <= 2
		a = get(args, 0) or 1
		s = get(args, 1) or 1.0
		return (erlang(a, scale=s), (0, inf), 
			{"shape" : a, "scale" : s})

	elif dname == "Exponential":
		assert len(args) <= 1
		s = get(args, 0) or 1.0
		return (expon(scale=s), (0, inf), {})

	elif dname == "FDist":
		assert len(args) == 2
		d1, d2 = args
		return (scipy.stats.f(d1, d2), (0, inf), {})

	elif dname == "Gamma":
		assert len(args) <= 2
		a = get(args, 0) or 1.0
		s = get(args, 1) or 1.0
		return (gamma(a, scale=s), (0, inf), 
			{"shape" : a, "scale" : s, "rate" : 1.0 / s})

	elif dname == "GeneralizedPareto":
		assert len(args) <= 3
		a = get(args, 0) or 1.0
		s = get(args, 1) or 1.0
		z = get(args, 2) or 0.0
		maxbnd = inf
		if a < 0.0:
			maxbnd = z - s / a
		return (genpareto(c=a, scale=s, loc=z), (z, maxbnd),
			{"shape" : a, "scale" : s, "location": z})

	elif dname == "Geometric":
		assert len(args) <= 1
		p = get(args, 0) or 0.5
		return (geom(p), (0, inf), 
			{"succprob" : p, "failprob" : 1.0 - p})

	elif dname == "Gumbel":
		assert len(args) <= 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (gumbel_r(l, s), (-inf, inf), 
			{"location" : l, "scale" : s})

	elif dname == "Hypergeometric":
		assert len(args) == 3
		ns, nf, n = [int(t) for t in args]
		return (hypergeom(ns + nf, ns, n), 
			(max(n - nf, 0), min(ns, n)), {})

	elif dname == "InverseGamma":
		assert len(args) <= 2
		a = get(args, 0) or 1.0
		s = get(args, 1) or 1.0
		return (invgamma(a, scale=s), (0, inf), 
			{"shape" : a, "scale" : s, "rate" : 1.0 / s})

	elif dname == "InverseGaussian":
		assert len(args) <= 2
		mu = get(args, 0) or 1.0
		lam = get(args, 1) or 1.0
		assert lam == 1.0  # scipy only supports this case
		return (invgauss(mu), (0, inf), {"shape" : lam})

	elif dname == "Laplace":
		assert len(args) <= 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (laplace(l, s), (-inf, inf), {"location" : l, "scale" : s})

	elif dname == "Logistic":
		assert len(args) <= 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (logistic(l, s), (-inf, inf), {"location" : l, "scale" : s})

	elif dname == "LogNormal":
		assert len(args) <= 2
		mu = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		assert mu == 0.0  # scipy only supports this case
		return (lognorm(s), (0, inf), {"meanlogx" : mu, "stdlogx" : s})

	elif dname == "NegativeBinomial":
		assert len(args) <= 2
		r = get(args, 0) or 1
		p = get(args, 1) or 0.5
		return (nbinom(r, p), (0, inf), 
			{"succprob" : p, "failprob" : 1.0 - p})

	elif dname == "Normal":
		assert len(args) == 2
		mu = args[0]
		sig = args[1]
		return (norm(mu, sig), (-inf, inf), {})

	elif dname == "NormalCanon":
		assert len(args) == 2
		h = args[0]
		J = args[1]
		return (norm(h/J, sqrt(1.0/J)), (-inf, inf), {})

	elif dname == "Pareto":
		assert len(args) <= 2
		a = get(args, 0) or 1.0
		s = get(args, 1) or 1.0
		return (pareto(a, scale=s), (s, inf), {"shape" : a, "scale" : s})

	elif dname == "Poisson":
		assert len(args) <= 1
		if len(args) == 0:
			lam = 1.0
		else:
			lam = get(args, 0)
		if lam == 0.0:
			upper = 0
		else:
			upper = inf
		return (poisson(lam), (0, upper), {"rate":lam})

	elif dname == "Rayleigh":
		assert len(args) <= 1
		s = get(args, 0) or 1.0
		return (rayleigh(scale=s), (0, inf), {})

	elif dname == "Skellam":
		assert len(args) <= 2
		if len(args) <= 1:
			u1 = u2 = get(args, 0) or 1.0
		else:
			u1, u2 = args
		return (skellam(u1, u2), (-inf, inf), {})

	elif dname == "SymTriangularDist":
		assert len(args) <= 2
		l = get(args, 0) or 0.0
		s = get(args, 1) or 1.0
		return (triang(0.5, loc=l-s, scale=s*2.0), (l-s, l+s), 
			{"location" : l, "scale" : s})

	elif dname == "TDist":
		assert len(args) == 1
		df = args[0]
		return (scipy.stats.t(df), (-inf, inf), {"dof" : df})

	elif dname == "TruncatedNormal":
		assert len(args) == 4
		mu, sig, a, b = args
		za = (a - mu) / sig
		zb = (b - mu) / sig
		za = max(za, -1000.0)
		zb = min(zb, 1000.0)
		return (truncnorm(za, zb, loc=mu, scale=sig), (a, b), {})

	elif dname == "TriangularDist":
		assert 2 <= len(args) and len(args) <= 3
		a = args[0]
		b = args[1]
		c = get(args, 2) or 0.5 * (a + b)
		return (triang((c - a) / (b - a), loc=a, scale=b-a), (a, b), 
			{"mode" : c})

	elif dname == "Uniform":
		assert len(args) <= 2
		if len(args) == 0:
			a, b = 0.0, 1.0
		elif len(args) == 1:
			a, b = 0.0, args[0]
		else:
			a, b = args
		return (uniform(a, b-a), (a, b), {"location" : a, "scale" : b - a})

	elif dname == "Weibull":
		assert len(args) <= 2
		a = get(args, 0) or 1.0
		s = get(args, 1) or 1.0
		return (weibull_min(a, scale=s), (0, inf), {"shape" : a, "scale" : s})

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
	try:
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
	except IndexError:
		# Poisson(0.0) will throw IndexError exception for entropy()
		if distr_name == "Poisson" and pdict["rate"] == 0.0:
			jdict = {"dtype" : distr_name,
			"params" : pdict,
			"minimum" : json_num(r_min),
			"maximum" : json_num(r_max),
			"mean" : json_num(0.0),
			"var" : json_num(0.0),
			"entropy" : json_num(0.0),
			"median" : json_num(0.0), 
			"q10" : json_num(0.0), 
			"q25" : json_num(0.0), 
			"q50" : json_num(0.0),
			"q75" : json_num(0.0),
			"q90" : json_num(0.0)}
		else:
			raise
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

	elif distr_name == "Poisson" and pdict["rate"] == 0.0:
		jdict["points"] = [{"x" : 0, "logpdf" : 0.0, "cdf" : 1.0}]

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


