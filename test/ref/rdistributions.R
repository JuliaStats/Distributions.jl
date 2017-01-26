# R6 classes for representing distributions in R

library(R6)

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
#  Auxiliary functions
#
#################################################

clamp <- function(a, b, x) {
    pmin(pmax(x, a), b)
}

xlogx <- function(x) {
    if (x > 0) { x * log(x) } else { 0.0 }
}

#################################################
#
#  Specific distributions
#
#################################################

# Arcsine

Arcsine <- R6Class("Arcsine",
    inherit = ContinuousDistribution,
    public = list(
        names = c("a", "b"),
        a = NA,
        b = NA,
        rd = NA,  # R distr object
        initialize = function(a, b) {
            self$a <- a
            self$b <- b
            self$rd <- distr::Arcsine() * ((b-a)/2) + ((a+b)/2)
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            list(location=self$a,
                 scale=self$b - self$a,
                 mean=(self$a + self$b) * 0.5,
                 var=1/8,
                 skewness=0,
                 kurtosis=-1.5,
                 median=(self$a + self$b) * 0.5,
                 entropy=log(pi/4) + log(self$b - self$a))
        },
        pdf = function(x, log=FALSE) { distr::d(self$rd)(x, log=log) },
        cdf = function(x) { distr::p(self$rd)(x) },
        quan = function(v) { distr::q(self$rd)(v) }
    )
)


# Bernoulli

Bernoulli <- R6Class("Bernoulli",
    inherit = DiscreteDistribution,
    public = list(
        names = c("p"),
        p = NA,
        initialize = function(p) {
            self$p <- p
        },
        supp = function() { c(0, 1) },
        properties = function() {
            p <- self$p
            q <- 1.0 - p
            list(succprob=p,
                 failprob=q,
                 mean=p,
                 median= if(p<=0.5) {0} else {1},
                 var=p * q,
                 skewness=(1 - 2*p) / sqrt(p*q),
                 kurtosis=(1 - 6*p*q) / (p*q),
                 entropy=-(xlogx(p) + xlogx(q)))
        },
        pdf = function(x, log=FALSE){ dbinom(x, 1, self$p, log=log) },
        cdf = function(x){ pbinom(x, 1, self$p) },
        quan = function(v){ qbinom(v, 1, self$p) }
    )
)

# Binomial

Binomial <- R6Class("Binomial",
    inherit = DiscreteDistribution,
    public = list(
        names = c("n", "p"),
        n = NA,
        p = NA,
        initialize = function(n, p) {
            self$n <- n
            self$p <- p
        },
        supp = function() { c(0, self$n) },
        properties = function() {
            n <- self$n
            p <- self$p
            q <- 1.0 - p
            list(succprob=p,
                 failprob=q,
                 ntrials=n,
                 mean=n * p,
                 median=round(n * p),
                 var=n * p * q,
                 skewness=(q - p) / sqrt(n*p*q),
                 kurtosis=(1 - 6*p*q) / (n*p*q))
        },
        pdf = function(x, log=FALSE) { dbinom(x, self$n, self$p, log=log) },
        cdf = function(x) { pbinom(x, self$n, self$p) },
        quan = function(v) { qbinom(v, self$n, self$p) }
    )
)

# Beta

Beta <- R6Class("Beta",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "beta"),
        alpha = NA,
        beta = NA,
        initialize = function(a, b) {
            self$alpha <- a
            self$beta <- b
        },
        supp = function() { c(0.0, 1.0) },
        properties = function() {
            a <- self$alpha
            b <- self$beta
            skew <- 2 * (b - a) * sqrt(a + b + 1) / (a + b + 2) / sqrt(a * b)
            kurt.num <- 6 * ((a - b)^2 * (a + b + 1) - a * b * (a + b + 2))
            kurt.den <- a * b * (a + b + 2) * (a + b + 3)
            ent <- lbeta(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) +
                   (a + b - 2) * digamma(a + b)
            list(mean=a / (a + b),
                 meanlogx=digamma(a) - digamma(a + b),
                 var=(a * b) / (a + b)^2 / (a + b + 1.0),
                 varlogx=trigamma(a) - trigamma(a + b),
                 skewness=skew,
                 kurtosis=kurt.num / kurt.den,
                 entropy=ent)
        },
        pdf = function(x, log=FALSE){ dbeta(x, self$alpha, self$beta, log=log) },
        cdf = function(x) { pbeta(x, self$alpha, self$beta) },
        quan = function(v) { qbeta(v, self$alpha, self$beta) }
    )
)

# Cauchy

Cauchy <- R6Class("Cauchy",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma"),
        mu = NA,
        sigma = NA,
        initialize = function(u, s) {
            self$mu <- u
            self$sigma <- s
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            list(location=self$mu,
                 scale=self$sigma,
                 median=self$mu,
                 entropy=log(self$sigma) + log(4 * pi))
        },
        pdf = function(x, log=FALSE) { dcauchy(x, self$mu, self$sigma, log=log) },
        cdf = function(x) { pcauchy(x, self$mu, self$sigma) },
        quan = function(v) { qcauchy(v, self$mu, self$sigma) }
    )
)

# Chisq

Chisq <- R6Class("Chisq",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu"),
        nu = NA,
        initialize = function(nu) {
            self$nu <- nu
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            k <- self$nu
            list(dof=k,
                 mean=k,
                 var=2 * k,
                 skewness=sqrt(8 / k),
                 kurtosis=12 / k,
                 entropy=k / 2 + log(2) + lgamma(k/2) + (1 - k/2) * digamma(k/2))
        },
        pdf = function(x, log=FALSE) { dchisq(x, self$nu, log=log) },
        cdf = function(x) { pchisq(x, self$nu) },
        quan = function(v) { qchisq(v, self$nu) }
    )
)

# DiscreteUniform

DiscreteUniform <- R6Class("DiscreteUniform",
    inherit = DiscreteDistribution,
    public = list(
        a = NA,
        b = NA,
        s = NA,
        names = c("a", "b"),
        initialize = function(a, b) {
            self$a <- a
            self$b <- b
            self$s <- b - a + 1
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            a <- self$a
            b <- self$b
            s <- self$s
            list(span=s,
                 probval=1/s,
                 mean=(a + b)/2,
                 median=(a + b)/2,
                 var=(s^2 - 1)/12,
                 skewness=0,
                 kurtosis=-(6 * (s^2 + 1))/(5 * (s^2 - 1)),
                 entropy=log(s))
        },
        pdf = function(x, log=FALSE) {
            a <- self$a
            b <- self$b
            s <- self$s
            t <- x >= a & x <= b
            if (log) {
                ifelse(t, -log(s), -Inf)
            } else {
                ifelse(t, 1 / s, 0.0)
            }
        },
        cdf = function(x) {
            a <- self$a
            b <- self$b
            clamp(0.0, 1.0, (x - a + 1) / (b - a + 1))
        },
        quan = function(v) {
            a <- self$a
            b <- self$b
            floor(a + (b - a + 1) * clamp(0.0, 1.0, v))
        }
    )
)

# Exponential

Exponential <- R6Class("Exponential",
    inherit = ContinuousDistribution,
    public = list(
        names = c("theta"),
        theta = NA,
        beta = NA,
        initialize = function(s) {
            self$theta <- s
            self$beta <- 1 / s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            s <- self$theta
            list(scale=s,
                 rate=self$beta,
                 mean=s,
                 median=s * log(2),
                 var=s^2,
                 skewness=2.0,
                 kurtosis=6.0,
                 entropy=1.0 + log(s))
        },
        pdf = function(x, log=FALSE) { dexp(x, self$beta, log=log) },
        cdf = function(x) { pexp(x, self$beta) },
        quan = function(v) { qexp(v, self$beta) }
    )
)

# FDist

FDist <- R6Class("FDist",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu1", "nu2"),
        nu1 = NA,
        nu2 = NA,
        initialize = function(nu1, nu2) {
            self$nu1 <- nu1
            self$nu2 <- nu2
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            d1 <- self$nu1
            d2 <- self$nu2
            var.num <- 2 * d2^2 * (d1 + d2 - 2)
            var.den <- d1 * (d2 - 2)^2 * (d2 - 4)
            skew.num <- (2 * d1 + d2 - 2) * sqrt(8 * (d2 - 4))
            skew.den <- (d2 - 6) * sqrt(d1 * (d1 + d2 - 2))
            list(mean = d2 / (d2 - 2),
                 var = var.num / var.den,
                 skewness = skew.num / skew.den)
        },
        pdf = function(x, log=FALSE) { df(x, self$nu1, self$nu2, log=log) },
        cdf = function(x) { pf(x, self$nu1, self$nu2) },
        quan = function(v) { qf(v, self$nu1, self$nu2) }
    )
)

# Gamma

Gammad <- R6Class("Gammad",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "theta"),
        alpha = NA,
        theta = NA,
        beta = NA,
        initialize = function(a, s) {
            self$alpha <- a
            self$theta <- s
            self$beta <- 1 / s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            a <- self$alpha
            s <- self$theta
            list(shape=a,
                 scale=s,
                 mean=a * s,                 
                 var=a * s^2,
                 skewness=2 / sqrt(a),
                 kurtosis=6 / a,
                 entropy=a + log(s) + lgamma(a) + (1 - a) * digamma(a)
            )
        },
        pdf = function(x, log=FALSE) { dgamma(x, self$alpha, self$beta, log=log) },
        cdf = function(x) { pgamma(x, self$alpha, self$beta) },
        quan = function(v) { qgamma(v, self$alpha, self$beta) }
    )
)

# Geometric

Geometric <- R6Class("Geometric",
    inherit = DiscreteDistribution,
    public = list(
        names = c("p"),
        p = NA,
        initialize = function(p) {
            self$p <- p
        },
        supp = function(){ c(0, Inf) },
        properties = function() {
            p <- self$p
            list(succprob=p,
                 failprob=1.0-p,
                 mean=(1-p)/p,
                 var=(1-p)/p^2,
                 skewness=(2-p)/sqrt(1-p),
                 kurtosis=6 + p^2/(1-p),
                 entropy=-((1-p)*log(1-p) + p * log(p))/p)
        },
        pdf = function(x, log=FALSE) { dgeom(x, self$p, log=log) },
        cdf = function(x) { pgeom(x, self$p) },
        quan = function(v){ qgeom(v, self$p) }
    )
)

# Hypergeometric

Hypergeometric <- R6Class("Hypergeometric",
    inherit = DiscreteDistribution,
    public = list(
        names = c("ns", "nf", "n"),
        ns = NA,
        nf = NA,
        n = NA,
        initialize = function(ns, nf, n) {
            self$ns <- ns
            self$nf <- nf
            self$n <- n
        },
        supp = function(){
            N <- self$ns + self$nf
            K <- self$ns
            n <- self$n
            c(max(0, n+K-N), min(n, K))
        },
        properties = function() {
            N <- self$ns + self$nf
            K <- self$ns
            n <- self$n
            mean.val <- n * (K / N)
            var.val  <- n * (K / N) * ((N-K) / N) * ((N - n) / (N - 1))
            skew.val <- ((N - 2*K) * sqrt(N - 1) * (N - 2*n)) /
                (sqrt(n * K * (N - K) * (N - n)) * (N - 2))
            kurt.num <- (N-1) * N^2 * (N * (N+1)- 6 * K * (N-K) - 6 * n * (N-n)) +
                6 * n * K * (N-K) * (N-n) * (5*N-6)
            kurt.den <- n * K * (N - K) * (N - n) * (N - 2) * (N - 3)
            kurt.val <- kurt.num / kurt.den
            list(mean=mean.val,
                 var=var.val,
                 skewness=skew.val,
                 kurtosis=kurt.val)
        },
        pdf = function(x, log=FALSE){ dhyper(x, self$ns, self$nf, self$n, log=log) },
        cdf = function(x){ phyper(x, self$ns, self$nf, self$n) },
        quan = function(v){ qhyper(v, self$ns, self$nf, self$n) }
    )
)

# NegativeBinomial

NegativeBinomial <- R6Class("NegativeBinomial",
    inherit = DiscreteDistribution,
    public = list(
        names = c("r", "p"),
        r = NA,
        p = NA,
        initialize = function(r, p) {
            self$r <- r
            self$p <- p
        },
        supp = function(){ c(0, Inf) },
        properties = function(){
            r <- self$r
            p <- self$p
            list(succprob=p,
                 failprob=1.0-p,
                 mean=(1-p) * r / p,
                 var=(1-p) * r / p^2,
                 skewness=(2-p) / sqrt((1-p) * r),
                 kurtosis=6 / r + p^2 / ((1-p)*r))
        },
        pdf = function(x, log=FALSE){ dnbinom(x, self$r, self$p, log=log)},
        cdf = function(x) { pnbinom(x, self$r, self$p) },
        quan = function(v) { qnbinom(v, self$r, self$p) }
    )
)

# Normal

Normal <- R6Class("Normal",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma"),
        mu = NA,
        sigma = NA,
        initialize = function(u, s) {
            self$mu <- u
            self$sigma <- s
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            list(mean=u,
                 var=s^2,
                 location=u,
                 scale=s,
                 skewness=0,
                 kurtosis=0,
                 entropy=(log(2 * pi) + 1 + log(s^2))/2)
        },
        pdf = function(x, log=FALSE) { dnorm(x, self$mu, self$sigma, log=log) },
        cdf = function(x) { pnorm(x, self$mu, self$sigma) },
        quan = function(v) { qnorm(v, self$mu, self$sigma ) }
    )
)

# Poisson

Poisson <- R6Class("Poisson",
    inherit = DiscreteDistribution,
    public = list(
        names = c("lambda"),
        lambda = NA,
        initialize = function(lambda) {
            self$lambda <- lambda
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            lam <- self$lambda
            list(rate=lam,
                 mean=lam,
                 var=lam,
                 skewness=1/sqrt(lam),
                 kurtosis=1/lam)
        },
        pdf = function(x, log=FALSE) { dpois(x, self$lambda, log=log) },
        cdf = function(x){ ppois(x, self$lambda) },
        quan = function(v){ qpois(v, self$lambda) }
    )
)

# TDist

TDist <- R6Class("TDist",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu"),
        nu = NA,
        initialize = function(nu) {
            self$nu <- nu
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            nu <- self$nu
            a <- (nu + 1) / 2
            ent.val <- a * (digamma(a) - digamma(nu / 2)) +
                       log(nu) / 2 + lbeta(nu / 2, 1 / 2)
            list(dof=nu,
                 mean=ifelse(nu > 1, 0, NaN),
                 median=0,
                 var=ifelse(nu > 2, nu / (nu - 2),
                     ifelse(nu > 1, Inf, NaN)),
                 skewness = ifelse(nu > 3, 0, NaN),
                 kurtosis = ifelse(nu > 4, 6 / (nu - 4),
                            ifelse(nu > 2, Inf, NaN)),
                 entropy = ent.val)
        },
        pdf = function(x, log=FALSE) { dt(x, self$nu, log=log) },
        cdf = function(x) { pt(x, self$nu) },
        quan = function(v) { qt(v, self$nu) }
    )
)

# Uniform

Uniform <- R6Class("Uniform",
    inherit = ContinuousDistribution,
    public = list(
        names = c("a", "b"),
        a = NA,
        b = NA,
        initialize = function(a, b) {
            self$a <- a
            self$b <- b
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            a <- self$a
            b <- self$b
            list(location = a,
                 scale = b - a,
                 mean = (a + b) / 2,
                 median = (a + b) / 2,
                 var = (b - a)^2 / 12,
                 skewness = 0,
                 kurtosis = -6 / 5,
                 entropy = log(b - a))
        },
        pdf = function(x, log=FALSE) { dunif(x, self$a, self$b, log=log) },
        cdf = function(x) { punif(x, self$a, self$b) },
        quan = function(v) { qunif(v, self$a, self$b) }
    )
)

# Weibull

Weibull <- R6Class("Weibull",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "theta"),
        alpha = NA,
        theta = NA,
        initialize = function(a, s) {
            self$alpha <- a
            self$theta <- s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            a <- self$alpha
            s <- self$theta
            var.val <- s^2 * (gamma(1 + 2 / a) - gamma(1 + 1 / a)^2)
            gv <- -digamma(1)
            list(shape = a,
                 scale = s,
                 mean = s * gamma(1.0 + 1 / a),
                 median = s * (log(2) ^ (1 / a)),
                 var = var.val,
                 entropy = gv * (1 - 1 / a) + log(s / a) + 1)
        },
        pdf = function(x, log=FALSE) { dweibull(x, self$alpha, self$theta, log=log) },
        cdf = function(x) { pweibull(x, self$alpha, self$theta) },
        quan = function(v) { qweibull(v, self$alpha, self$theta) }
    )
)
