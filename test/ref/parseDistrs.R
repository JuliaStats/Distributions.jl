# Parse distribution from a list entry

source("rdistributions.R")

read.entries <- function(filename) {
    # Read a list of entries from a given file
    lst <- read.table(filename, header=FALSE,
        sep="\t", blank.lines.skip = TRUE, comment.char="#")
    as.vector(lst$V1)
}

parse.entry <- function(entry) {
    # Parse an entry into a distribution name, and
    # a vector of argument values

    terms <- unlist(strsplit(entry, "\\s*,\\s*|\\(|\\)"))
    nt <- length(terms)
    list(name=terms[1],
         args=as.numeric(tail(terms, nt-1)))
}


get.distr <- function(entry) {
    # Get a distribution object based on a given entry
    parsed <- parse.entry(entry)
    dname <- parsed$name
    dargs <- parsed$args

    # parse arguments
    if (length(dargs) == 1 && is.na(dargs)) {
        nargs <- 0
    } else {
        nargs <- length(dargs)
        a1 <- dargs[1]
        a2 <- ifelse(nargs >= 2, dargs[2], NA)
        a3 <- ifelse(nargs >= 3, dargs[3], NA)
    }

    distr <- switch (dname,
        Arcsine = {
                 if (nargs == 0) { Arcsine$new(0, 1) }
            else if (nargs == 1) { Arcsine$new(0, a1) }
            else if (nargs == 2) { Arcsine$new(a1, a2) }
        },
        Bernoulli = {
                 if (nargs == 0) { Bernoulli$new(0.5) }
            else if (nargs == 1) { Bernoulli$new(a1) }
        },
        Beta = {
                 if (nargs == 0) { Beta$new(1, 1) }
            else if (nargs == 1) { Beta$new(a1, a1) }
            else if (nargs == 2) { Beta$new(a1, a2) }
        },
        Binomial = {
                 if (nargs == 0) { Binomial$new(1, 0.5) }
            else if (nargs == 1) { Binomial$new(a1, 0.5) }
            else if (nargs == 2) { Binomial$new(a1, a2) }
        },
        Cauchy = {
                 if (nargs == 0) { Cauchy$new(0, 1) }
            else if (nargs == 1) { Cauchy$new(a1, 1) }
            else if (nargs == 2) { Cauchy$new(a1, a2) }
        },
        Chisq = {
            if (nargs == 1) { Chisq$new(a1) }
        },
        DiscreteUniform = {
                 if (nargs == 0) { DiscreteUniform$new(0, 1) }
            else if (nargs == 1) { DiscreteUniform$new(0, a1) }
            else if (nargs == 2) { DiscreteUniform$new(a1, a2) }
        },
        Exponential = {
                 if (nargs == 0) { Exponential$new(1) }
            else if (nargs == 1) { Exponential$new(a1) }
        },
        Erlang = {
                 if (nargs == 0) { Gammad$new(1, 1) }
            else if (nargs == 1) { Gammad$new(a1, 1) }
            else if (nargs == 2) { Gammad$new(a1, a2) }
        },
        FDist = {
            if (nargs == 2) { FDist$new(a1, a2) }
        },
        Gamma = {
                 if (nargs == 0) { Gammad$new(1, 1) }
            else if (nargs == 1) { Gammad$new(a1, 1) }
            else if (nargs == 2) { Gammad$new(a1, a2) }
        },
        Geometric = {
                 if (nargs == 0) { Geometric$new(0.5) }
            else if (nargs == 1) { Geometric$new(a1) }
        },
        Hypergeometric = {
            if (nargs == 3) { Hypergeometric$new(a1, a2, a3) }
        },
        NegativeBinomial = {
                 if (nargs == 0) { NegativeBinomial$new(1, 0.5) }
            else if (nargs == 1) { NegativeBinomial$new(a1, 0.5) }
            else if (nargs == 2) { NegativeBinomial$new(a1, a2) }
        },
        Normal = {
                 if (nargs == 0) { Normal$new(0, 1) }
            else if (nargs == 1) { Normal$new(a1, 1) }
            else if (nargs == 2) { Normal$new(a1, a2) }
        },
        NormalCanon = {
                 if (nargs == 0) { Normal$new(0, 1) }
            else if (nargs == 2) { Normal$new(a1 / a2, 1 / sqrt(a2)) }
        },
        Poisson = {
                 if (nargs == 0) { Poisson$new(1) }
            else if (nargs == 1) { Poisson$new(a1) }
        },
        Skellam = {
                 if (nargs == 0) { Skellam$new(1, 1) }
            else if (nargs == 1) { Skellam$new(a1, a1) }
            else if (nargs == 2) { Skellam$new(a1, a2) }
        },
        TDist = {
            if (nargs == 1) { TDist$new(a1) }
        },
        Uniform = {
                 if (nargs == 0) { Uniform$new(0, 1) }
            else if (nargs == 2) { Uniform$new(a1, a2) }
        },
        Weibull = {
                 if (nargs == 0) { Weibull$new(1, 1) }
            else if (nargs == 1) { Weibull$new(a1, 1) }
            else if (nargs == 2) { Weibull$new(a1, a2) }
        }
    )

    if (is.null(distr)) {
        stop(paste("Unrecognized distribution:", dname))
    }
    return(distr)
}
