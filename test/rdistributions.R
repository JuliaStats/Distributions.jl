# S4 classes for representing distributions in R

library(methods)

#################################################
#
#  Declare generic functions
#
#################################################

setGeneric("properties",
    function(object) standardGeneric("properties"))

setGeneric("supp",
    function(object) standardGeneric("supp"))

setGeneric("pd",
    function(object, x) standardGeneric("pd"))

setGeneric("logpd",
    function(object, x) standardGeneric("logpd"))

setGeneric("cd",
    function(object, x) standardGeneric("cd"))

setGeneric("quan",
    function(object, v) standardGeneric("quan"))


setDistr <- function(className,
        supp=NULL, properties=NULL,
        pd=NULL, logpd=NULL, cd=NULL, quan=NULL) {

    setMethod("supp", className, supp)
    setMethod("properties", className, properties)
    setMethod("pd", className, pd)
    setMethod("logpd", className, logpd)
    setMethod("cd", className, cd)
    setMethod("quan", className, quan)
}

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

setClass("Bernoulli",
    representation(p="numeric"))

setDistr("Bernoulli",
    supp=function(object){ c(0, 1) },
    properties=function(object) {
        p <- object@p
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
    pd=function(object, x){ dbinom(x, 1, object@p) },
    logpd=function(object, x){ dbinom(x, 1, object@p, log=TRUE) },
    cd=function(object, x){ pbinom(x, 1, object@p) },
    quan=function(object, v){ qbinom(v, 1, object@p) }
)


# Binomial

setClass("Binomial",
    representation(n="numeric", p="numeric"))

setDistr("Binomial",
    supp=function(object){ c(0, objecr@n) },
    properties=function(object) {
        n <- object@n
        p <- object@p
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
    pd=function(object, x){ dbinom(x, object@n, object@p) },
    logpd=function(object, x){ dbinom(x, object@n, object@p, log=TRUE) },
    cd=function(object, x){ pbinom(x, object@n, object@p) },
    quan=function(object, v){ qbinom(v, object@n, object@p) }
)

# DiscreteUniform

setClass("DiscreteUniform",
    representation(a="numeric", b="numeric"))

setDistr("DiscreteUniform",
    supp=function(object){ c(object@a, objecr@b) },
    properties=function(object) {
        a <- object@a
        b <- object@b
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
    pd=function(object, x) {
        a <- object@a
        b <- object@b
        pv <- 1 / (b - a + 1)
        ifelse(x >= a & x <= b, pv, 0.0)
    },
    logpd=function(object, x) {
        a <- object@a
        b <- object@b
        lpv <- -log(b - a + 1)
        ifelse(x >= a & x <= b, lpv, -Inf)
    },
    cd=function(object, x) {
        a <- object@a
        b <- object@b
        clamp(0.0, 1.0, (x - a + 1) / (b - a + 1))
    },
    quan=function(object, v) {
        a <- object@a
        b <- object@b
        floor(a + (b - a) * clamp(0.0, 1.0, v))
    }
)

# Geometric

setClass("Geometric",
    representation(p="numeric"))

setDistr("Geometric",
    supp=function(object){ c(0, Inf) },
    properties=function(object) {
        p <- object@p
        list(succprob=p,
             failprob=1.0-p,
             mean=(1-p)/p,
             var=(1-p)/p^2,
             skewness=(2-p)/sqrt(1-p),
             kurtosis=6 + p^2/(1-p),
             entropy=-((1-p)*log2(1-p) + p * log2(p))/p)
    },
    pd=function(object, x){ dgeom(x, object@p) },
    logpd=function(object, x){ dgeom(x, object@p, log=TRUE) },
    cd=function(object, x){ pgeom(x, object@p) },
    quan=function(object, v){ qgeom(v, object@p) }
)

# Hypergeometric

setClass("Hypergeometric",
    representation(ns="numeric", nf="numeric", n="numeric"))

setDistr("Hypergeometric",
    supp=function(object){
        N <- object@ns + object@nf
        K <- object@ns
        n <- object@n
        c(max(0, n+K-N), min(n, K))
    },
    properties=function(object) {
        N <- object@ns + object@nf
        K <- object@ns
        n <- object@n
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
    pd=function(object, x){
        dhyper(x, object@ns, object@nf, object@n)
    },
    logpd=function(object, x){
        dhyper(x, object@ns, object@nf, object@n, log=TRUE)
    },
    cd=function(object, x){
        phyper(x, object@ns, object@nf, object@n)
    },
    quan=function(object, v){
        qhyper(v, object@ns, object@nf, object@n)
    }
)

# NegativeBinomial

setClass("NegativeBinomial",
    representation(r="numeric", p="numeric"))

setDistr("NegativeBinomial",
    supp=function(object){ c(0, Inf) },
    properties=function(object){
        r <- object@r
        p <- object@p
        list(succprob=p,
             failprob=1.0-p,
             mean=p * r / (1-p),
             var=p * r / (1-p)^2,
             skewness=(1 + p) / sqrt(p * r),
             kurtosis=6 / r + (1-p)^2 / (p*r))
    },
    pd=function(object, x){ dnbinom(x, object@r, object@p)},
    logpd=function(object, x){ dnbinom(x, object@r, object@p, log=TRUE)},
    cd=function(object, x){ pnbinom(x, object@r, object@p) },
    quan=function(object, v){ qnbinom(v, object@r, object@p) }
)

# Poisson

setClass("Poisson",
    representation(lambda="numeric"))

setDistr("Poisson",
    supp=function(object){ c(0, Inf) },
    properties=function(object) {
        lam <- object@lambda
        list(rate=lam,
             mean=lam,
             var=lam,
             skewness=1/sqrt(lam),
             kurtosis=1/lam)
    },
    pd=function(object, x){ dpois(x, object@lambda) },
    logpd=function(object, x){ dpois(x, object@lambda, log=TRUE) },
    cd=function(object, x){ ppois(x, object@lambda) },
    quan=function(object, v){ qpois(v, object@lambda) }
)
