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
#  Discrete distributions
#
#################################################

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
                 median= if(p<0.5) {0} else if (p > 0.5) {1} else {0.5},
                 var=p * q,
                 skewness=(1 - 2*p) / sqrt(p*q),
                 kurtosis=(1 - 6*p*q) / (p*q),
                 entropy=-(xlogx(p) + xlogx(q)))
        },
        pdf = function(x){ dbinom(x, 1, self$p) },
        logpdf = function(x){ dbinom(x, 1, self$p, log=TRUE) },
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
                 median=floor(n * p),
                 var=n * p * q,
                 skewness=(q - p) / sqrt(n*p*q),
                 kurtosis=(1 - 6*p*q) / (n*p*q))
        },
        pdf = function(x) { dbinom(x, self$n, self$p) },
        logpdf = function(x) { dbinom(x, self$n, self$p, log=TRUE) },
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
                   (a - b + 2) * digamma(a + b)
            list(mean=a / (a + b),
                 meanlogx=digamma(a) - digamma(a + b),
                 var=(a * b) / (a + b)^2 / (a + b + 1.0),
                 varlogx=trigamma(a) - trigamma(a + b),
                 skewness=skew,
                 kurtosis=kurt.num / kurt.den,
                 entropy=ent)
        },
        pdf = function(x){ dbeta(x, self$alpha, self$beta) },
        logpdf = function(x){ dbeta(x, self$alpha, self$beta, log=TRUE) },
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
        pdf = function(x) { dcauchy(x, self$mu, self$sigma) },
        logpdf = function(x) { dcauchy(x, self$mu, self$sigma, log=TRUE) },
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
        pdf = function(x) { dchisq(x, self$nu) },
        logpdf = function(x) { dchisq(x, self$nu, log=TRUE) },
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
        names = c("a", "b"),
        initialize = function(a, b) {
            self$a <- a
            self$b <- b
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            a <- self$a
            b <- self$b
            s <- b - a + 1
            list(span=s,
                 probval=1/s,
                 mean=(a + b)/2,
                 median=(a + b)/2,
                 var=(s^2 - 1)/12,
                 skewness=0,
                 kurtosis=-(6 * (s^2 + 1))/(5 * (s^2 - 1)),
                 entropy=log(s))
        },
        pdf = function(x) {
            a <- self$a
            b <- self$b
            pv <- 1 / (b - a + 1)
            ifelse(x >= a & x <= b, pv, 0.0)
        },
        logpdf = function(x) {
            a <- self$a
            b <- self$b
            lpv <- -log(b - a + 1)
            ifelse(x >= a & x <= b, lpv, -Inf)
        },
        cdf = function(x) {
            a <- self$a
            b <- self$b
            clamp(0.0, 1.0, (x - a + 1) / (b - a + 1))
        },
        quan = function(v) {
            a <- self$a
            b <- self$b
            floor(a + (b - a) * clamp(0.0, 1.0, v))
        }
    )
)

# Exponential

Exponential <- R6Class("Exponential",
    inherit = ContinuousDistribution,
    public = list(
        names = c("theta"),
        theta = NA,
        initialize = function(s) {
            self$theta = s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            s <- self$theta
            list(scale=s,
                 rate=1 / s,
                 mean=s,
                 median=s * log(2),
                 var=s^2,
                 skewness=2.0,
                 kurtosis=6.0,
                 entropy=1.0 + log(s))
        },
        pdf = function(x) { dexp(x, 1.0 / self$theta) },
        logpdf = function(x) { dexp(x, 1.0 / self$theta, log=TRUE) },
        cdf = function(x) { pexp(x, 1.0 / self$theta) },
        quan = function(v) { qexp(v, 1.0 / self$theta) }
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
                 entropy=-((1-p)*log2(1-p) + p * log2(p))/p)
        },
        pdf = function(x) { dgeom(x, self$p) },
        logpdf = function(x) { dgeom(x, self$p, log=TRUE) },
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
        pdf = function(x){
            dhyper(x, self$ns, self$nf, self$n)
        },
        logpdf = function(x){
            dhyper(x, self$ns, self$nf, self$n, log=TRUE)
        },
        cdf = function(x){
            phyper(x, self$ns, self$nf, self$n)
        },
        quan = function(v){
            qhyper(v, self$ns, self$nf, self$n)
        }
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
                 mean=p * r / (1-p),
                 var=p * r / (1-p)^2,
                 skewness=(1 + p) / sqrt(p * r),
                 kurtosis=6 / r + (1-p)^2 / (p*r))
        },
        pdf = function(x){ dnbinom(x, self$r, self$p)},
        logpdf = function(x) { dnbinom(x, self$r, self$p, log=TRUE)},
        cdf = function(x) { pnbinom(x, self$r, self$p) },
        quan = function(v) { qnbinom(v, self$r, self$p) }
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
        pdf = function(x) { dpois(x, self$lambda) },
        logpdf = function(x){ dpois(x, self$lambda, log=TRUE) },
        cdf = function(x){ ppois(x, self$lambda) },
        quan = function(v){ qpois(v, self$lambda) }
    )
)
