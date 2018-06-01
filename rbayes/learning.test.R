# data generation.
LV3 = c("a", "b", "c")

a = sample(LV3, 5000, prob = rep(1/3, 3), replace = TRUE)
c = sample(LV3, 5000, prob = c(0.75, 0.2, 0.05), replace = TRUE)
f = sample(c("a", "b"), 5000, prob = rep(1/2, 2), replace = TRUE)

b = a
b[b == "a"] = sample(LV3, length(which(b == "a")), prob = c(0.8, 0.1, 0.1), replace = TRUE)
b[b == "b"] = sample(LV3, length(which(b == "b")), prob = c(0.4, 0.2, 0.4), replace = TRUE)
b[b == "c"] = sample(LV3, length(which(b == "c")), prob = c(0.1, 0.1, 0.8), replace = TRUE)

d = apply(cbind(a,c), 1, paste, collapse= ":")
d[d == "a:a"] = sample(LV3, length(which(d == "a:a")), prob = c(0.8, 0.1, 0.1), replace = TRUE)
d[d == "a:b"] = sample(LV3, length(which(d == "a:b")), prob = c(0.2, 0.1, 0.7), replace = TRUE)
d[d == "a:c"] = sample(LV3, length(which(d == "a:c")), prob = c(0.4, 0.2, 0.4), replace = TRUE)
d[d == "b:a"] = sample(LV3, length(which(d == "b:a")), prob = c(0.1, 0.8, 0.1), replace = TRUE)
d[d == "b:b"] = sample(LV3, length(which(d == "b:b")), prob = c(0.9, 0.05, 0.05), replace = TRUE)
d[d == "b:c"] = sample(LV3, length(which(d == "b:c")), prob = c(0.3, 0.4, 0.3), replace = TRUE)
d[d == "c:a"] = sample(LV3, length(which(d == "c:a")), prob = c(0.1, 0.1, 0.8), replace = TRUE)
d[d == "c:b"] = sample(LV3, length(which(d == "c:b")), prob = c(0.25, 0.5, 0.25), replace = TRUE)
d[d == "c:c"] = sample(LV3, length(which(d == "c:c")), prob = c(0.15, 0.45, 0.4), replace = TRUE)

e = apply(cbind(b,f), 1, paste, collapse= ":")
e[e == "a:a"] = sample(LV3, length(which(e == "a:a")), prob = c(0.8, 0.1, 0.1), replace = TRUE)
e[e == "a:b"] = sample(LV3, length(which(e == "a:b")), prob = c(0.4, 0.5, 0.1), replace = TRUE)
e[e == "b:a"] = sample(LV3, length(which(e == "b:a")), prob = c(0.2, 0.2, 0.6), replace = TRUE)
e[e == "b:b"] = sample(LV3, length(which(e == "b:b")), prob = c(0.3, 0.4, 0.3), replace = TRUE)
e[e == "c:a"] = sample(LV3, length(which(e == "c:a")), prob = c(0.1, 0.1, 0.8), replace = TRUE)
e[e == "c:b"] = sample(LV3, length(which(e == "c:b")), prob = c(0.25, 0.5, 0.25), replace = TRUE)

learning.test = data.frame(
  A = factor(a, levels = LV3),
  B = factor(b, levels = LV3),
  C = factor(c, levels = LV3),
  D = factor(d, levels = LV3),
  E = factor(e, levels = LV3),
  F = factor(f, levels = c("a", "b"))
)

# network specification.
dag = model2network("[A][C][F][B|A][D|A:C][E|B:F]")

bn = custom.fit(dag, list(
  A = matrix(rep(1/3, 3), ncol = 3, dimnames = list(NULL, LV3)),
  B = matrix(c(0.8, 0.1, 0.1, 0.4, 0.2, 0.4, 0.1, 0.1, 0.8), ncol = 3,
                 dimnames = list(B = LV3, A = LV3)),
  C = matrix(c(0.75, 0.2, 0.05), ncol = 3, dimnames = list(NULL, LV3)),
  D = array(c(0.8, 0.1, 0.1, 0.2, 0.1, 0.7, 0.4, 0.2, 0.4, 0.1, 0.8, 0.1, 0.9, 0.05, 0.05,
               0.3, 0.4, 0.3, 0.1, 0.1, 0.8, 0.25, 0.5, 0.25, 0.15, 0.45, 0.4), dim = c(3, 3, 3),
             dimnames = list(D = LV3, A = LV3, C = LV3)),
  E = array(c(0.8, 0.1, 0.1, 0.4, 0.5, 0.1, 0.2, 0.2, 0.6, 0.3, 0.4, 0.3, 0.1, 0.1, 0.8,
               0.25, 0.5, 0.25), dim = c(3, 3, 2), dimnames = list(E = LV3, B = LV3, F = c("a", "b"))),
  F = matrix(rep(1/2, 2), ncol = 2, dimnames = list(NULL, c("a", "b")))
))

