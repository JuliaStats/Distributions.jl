# R6 classes for representing distributions in R

library(R6)
library(extraDistr)

#################################################
#
#  Base classes
#
#################################################

DiscreteDistribution <- R6Class("DiscreteDistribution",
    public = list(is.discrete=TRUE)
)

ContinuousDistribution <- R6Class("ContinuousDistribution",
    public = list(is.discrete=FALSE)
)

#################################################
#
#  Discrete distributions
#
#################################################

source("discrete/bernoulli.R")
source("discrete/binomial.R")
source("discrete/betabinomial.R")
source("discrete/discreteuniform.R")
source("discrete/geometric.R")
source("discrete/hypergeometric.R")
source("discrete/negativebinomial.R")
source("discrete/noncentralhypergeometric.R")
source("discrete/poisson.R")
source("discrete/skellam.R")

#################################################
#
#  Continuous distributions
#
#################################################

source("continuous/arcsine.R")
source("continuous/beta.R")
source("continuous/betaprime.R")
source("continuous/cauchy.R")
source("continuous/chi.R")
source("continuous/chisq.R")
source("continuous/cosine.R")
source("continuous/exponential.R")
source("continuous/fdist.R")
source("continuous/frechet.R")
source("continuous/gamma.R")
source("continuous/generalizedextremevalue.R")
source("continuous/generalizedpareto.R")
source("continuous/gumbel.R")
source("continuous/inversegamma.R")
source("continuous/inversegaussian.R")
source("continuous/johnsonsu.R")
source("continuous/kumaraswamy.R")
source("continuous/laplace.R")
source("continuous/levy.R")
source("continuous/lindley.R")
source("continuous/logistic.R")
source("continuous/lognormal.R")
source("continuous/noncentralbeta.R")
source("continuous/noncentralchisq.R")
source("continuous/noncentralf.R")
source("continuous/noncentralt.R")
source("continuous/normal.R")
source("continuous/normalinversegaussian.R")
source("continuous/pareto.R")
source("continuous/pgeneralizedgaussian.R")
source("continuous/rayleigh.R")
source("continuous/rician.R")
source("continuous/studentizedrange.R")
source("continuous/tdist.R")
source("continuous/triangulardist.R")
source("continuous/truncatednormal.R")
source("continuous/uniform.R")
source("continuous/vonmises.R")
source("continuous/weibull.R")
