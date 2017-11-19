---
title: 'Maximum Likelihood and Bayes Estimation'
date: 2016-01-29
permalink: /posts/2016/01/MSE_and_Bayes_estimation/
tag:
  - machine learning
use_math: true
---

{% include toc %}

<div style="display:none">
$$\DeclareMathOperator{\E}{E}$$
$$\DeclareMathOperator{\KL}{KL}$$
$$\DeclareMathOperator{\Var}{Var}$$
$$\DeclareMathOperator{\Bias}{Bias}$$
</div>
As we know, maximum likelihood estimation (MLE) and Bayes estimation (BE) are two kinds of methods for parameter estimation in machine learning. However, they are on behalf of different view but closely interconnected with each other. In this article, I would like to talk about the differences and connections of them.

## Maximum Likelihood Estimation

Consider a set of $N$ examples $\mathscr{X}=\\{x^{(1)},\ldots, x^{(N)}\\}$ drawn independently from the true but unknown data generating distribution $p_{data}(x)$.  Let $p_{model}(x; \theta)$ be a parametric family of probability distributions over the same space indexed by $\theta$.  In other words, $p_{model}(x;\theta)$ maps any $x$ to a real number estimating the true probability $p_{data}(x)$.

The maximum likelihood estimator for $\theta$ is then defined by

$$
\theta_{ML} = \mathop{\arg\max}_\theta p_{model}(\mathscr{X};\theta) = \mathop{\arg\max}_\theta \prod_{i=1}^N p_{model}(x^{(i)};\theta)
$$

For convenience, we usually maximize the logarithm of that

$$
\theta_{ML} = \mathop{\arg\max}_\theta \sum_{i=1}^N \log p_{model}(x^{(i)};\theta)
$$

Since rescaling the cost function does not change the result of $\mathop{\arg\max}$, so we can divide by $N$ to obtain a formula expressed as an expectation.

$$
\theta_{ML} = \mathop{\arg\max}_\theta \E_{x\sim \hat{p}_{data}}\log p_{model}(x;\theta)
$$

Maximizing something is equivalent to minimizing the negative of something, thus we have

$$
\theta_{ML} = \mathop{\arg\min}_\theta -\E_{x\sim \hat{p}_{data}}\log p_{model}(x;\theta)
$$

One way to interprete MLE is to view what we are minimizing the dissimilarity between the experical distribution defined by training set and the model distribution, with the degree of dissimilarity between the two distributions measured by the KL divergence. The KL divergence is given by

$$
\KL(\hat{p}_{data}||p_{model})=\E_{x\sim \hat{p}_{data}}(\log \hat{p}_{data}(x) - \log p_{model}(x;\theta)).
$$

Since the expectation of $\log \hat{p}_{data}(x)$ is a constant, we can see the optimal $\theta$ of maximum likelihood principle attempts to minimize the KL divergence.

## Bayes Estimation

As discussed above, the frequentist perspective is the true parameter $\theta$ is fixed but unknown, while the MLE $\theta_{ML}$
is a random variable on account of it being a function of the data. But the bayesian perspective on statistics is quite different.
The data is intuitively observed rather than viewed randomly. They use prior probability distribution $p(\theta)$ to reflect some
knowledge they know about the distribution to some degree. Now that we have observed a set of data samples
$\mathscr{X}=\{x^{(1)},\ldots,x^{(N)}\}$, we can recover possibility or our belief about a certain value $\theta$ by combining
the prior with the conditional distribution $p(\mathscr{X}|\theta)$ via bayes formula

$$
p(\theta|\mathscr{X}) = {p(\mathscr{X}|\theta)p(\theta)\over p(\mathscr{X})},
$$

which is the posterior probability.

Unlike what we did in MLE, Bayes estimation was effected with respect to a full distribution over $\theta$.
The quintessential idea of bayes estimation is minimizing conditional risk or expected loss function $R(\hat{\theta}|X)$, given by

$$
R(\hat{\theta}|X) = \int_\Theta \lambda(\hat{\theta},\theta)p(\theta|X)d\theta,
$$

where $\Theta$ is the parameter space of $\theta$. If we take the loss function to be quadratic function, i.e. $\lambda(\hat{\theta},\theta)=(\theta-\hat{\theta})^2$, then the bayes estimation of $\theta$ is

$$
\theta_{BE} = \E(\theta|X) = \int_\Theta \theta p(\theta|X)d\theta.
$$

The proof is easy.

It is worth mentioning that in bayes learning, we need not to estimate $\theta$. Instead, we could give the probability distribution function of a sample $x$ directly. For example, after obsering $N$ data samples, the predicted distribution of the next example $x^{(N+1)}$, is given by

$$
p(x^{(N+1)}|\mathscr{X}) = \int p(x^{(N+1)}|\theta)p(\theta|\mathscr{X})d\theta.
$$

## Maximum A Posteriori Estimation

A more commonn way to estimate parameters is ccarried out using a so called maximum a posteriori (MAP) method. The MAP estimate choose the point of maximal posterior probability

$$
\theta_{MAP} = \mathop{\arg\max}_\theta p(\theta|\mathscr{X}) = \mathop{\arg\max}_\theta \log p(\mathscr{X}|\theta) + \log p(\theta)
$$

## Relations

As we talked above, Maximizing likelihood function is equivalent to minimizing the KL divergence between model distribution and empirical distribution. In a bayesian view of this, we can say that MLE is equivalent to minimizing empirical risk when the loss function is taken to be the logarithm loss (cross entropy loss).

The advatage brought by introducing the influence of the prior on MAP estimate is to leverage the additional information other than the unpredicted data. This additional information helps us to reduce the variance in MAP point estimate in comparison to MLE, however at the expense of increasing the bias. A good example help illustrate this idea.

**Example: (Linear Regression)**   The problem is to find appropriate $w$ such that a mapping defined by
$$
y=w^T x
$$ gives the best prediction of $y$ over the entire training set $\mathscr{X}=\\{x^{(1)}, \ldots,x^{(N)}\\}$. Expressing the predition in a matrix form,
$$
y= \mathscr{X}^T w
$$ Besides, let us asssume the conditional distribution of $y$ given $w$ and $\mathscr{X}$ is Gaussian distribution parametrized by mean vector $\mathscr{X}^T w$ and variance matrix $I$.

In this case, the MLE gives an estimate

$$
\hat{w}_{ML} = (\mathscr{X}^T\mathscr{X})^{-1}\mathscr{X}y. \tag 1 \label{eq-1}
$$

We also assume the prior of $w$ is another Gaussian distribution parametrized by mean $0$ and variance matrix $\Lambda_0=\lambda_0I$. With the prior specified, we can now determine the posterior distribution over the model parameters.

$$
\begin{align*}
p(w|\mathscr{X},y) &\propto p(y|\mathscr{X},w)p(w)\\
&\propto \exp\left(-{1\over2}(y-\mathscr{X}w)^T(y-\mathscr{X}w)\right)\exp\left(-{1\over2}w^T\Lambda_0^{-1}w\right)\\
&\propto \exp\left(-{1\over2}\left(-2y^T\mathscr{X}w + w^T\mathscr{X}^T\mathscr{X}w + w^T\Lambda_0^{-1}w\right)\right)\\
&\propto \exp\left(-{1\over2}(w-\mu_N)^T\Lambda_N^{-1}(w-\mu_N)\right).
\end{align*}
$$

where

$$
\begin{align*}
\Lambda_N &= (\mathscr{X}^T\mathscr{X} + \Lambda_0^{-1})^{-1}\\
\mu_N &= \Lambda_N\mathscr{X}^Ty
\end{align*}
$$

Thus the MAP estimate of the $w$ becomes

$$
\hat{w}_{MAP} = (\mathscr{X}^T\mathscr{X} + \lambda_0^{-1}I)^{-1}\mathscr{X}^T y. \tag 2 \label{eq-2}
$$

Compared \eqref{eq-2} with \eqref{eq-1}, we see that the MAP estimate amounts to adding a weighted term related with variance of prior distribution in the parenthesis at the basis of MLE. Also, it is easy to show that the MLE is unbiased, i.e. $\E(\hat{w}_{ML})=w$ and that it has a variance given by

$$
\Var(\hat{w}_{ML})=(\mathscr{X}^T\mathscr{X})^{-1}. \tag 3\label{eq-3}
$$

In order to derive the bias of the MAP estimate, we need to evaluate the expectation

$$
\begin{align*}
\E(\hat{w}_{MAP}) &= E(\Lambda_N \mathscr{X}^Ty)\\
&= \E(\Lambda_N \mathscr{X}^T(\mathscr{X}w + \epsilon))\\
&= \Lambda_N(\mathscr{X}^T\mathscr{X}w) + \Lambda_N\mathscr{X}^T \E(\epsilon)\\
&= (\mathscr{X}^T\mathscr{X} + \lambda_0^{-1}I)^{-1}\mathscr{X}^T\mathscr{X}w\\
&= (\mathscr{X}^T\mathscr{X} + \lambda_0^{-1}I)^{-1} (\mathscr{X}^T\mathscr{X} + \lambda_0^{-1}I - \lambda_0^{-1}I)w\\
&= (I - (\lambda_0\mathscr{X}^T\mathscr{X} + I)^{-1} )w
\end{align*}
$$

Thus, the bias can be derived as

$$
\Bias(\hat{w}_{MAP}) = \E(\hat{w}_{MAP}) - w = -(\lambda_0\mathscr{X}^T\mathscr{X} + I)^{-1}w.
$$

Therefore, we can conclude that the MAP estimate is unbiased, and as the variance of prior $\lambda_0 \to \infty$, the bias tends to $0$. And as the variance of the prior $\lambda_0 \to 0$, the bias tends to $w$. This case is exactly the ML estimate, because the variance tending to $\infty$ implies that the prior distribution is asymptotically uniform. In other words, knowing nothing about the prior distribution, we assign the same probability to every value of $w$.

Before computing the variance, we need to compute

$$
\begin{align*}
\E(\hat{w}_{MAP}\hat{w}_{MAP}^T) &= \E(\Lambda_N \mathscr{X}^T yy^T \mathscr{X} \Lambda_N)\\
&= \E(\Lambda_N \mathscr{X}^T (\mathscr{X}w+\epsilon)(\mathscr{X}w+\epsilon)^T \mathscr{X} \Lambda_N) \\
&= \Lambda_N \mathscr{X}^T\mathscr{X}ww^T\mathscr{X}^T\mathscr{X}\Lambda_N + \Lambda_N \mathscr{X}^T\E(\epsilon\epsilon^T)\mathscr{X}\Lambda_N \\
&= \Lambda_N \mathscr{X}^T\mathscr{X}ww^T\mathscr{X}^T\mathscr{X}\Lambda_N + \Lambda_N \mathscr{X}^T\mathscr{X}\Lambda_N \\
&= \E(\hat{w}_{MAP})\E(\hat{w}_{MAP})^T + \Lambda_N \mathscr{X}^T\mathscr{X}\Lambda_N.
\end{align*}
$$

Therefore, the variance of the MAP estimate of our linear regression model is given by

$$
\begin{align*}
\Var(\hat{w}_{MAP}) &= \E(\hat{w}_{MAP}\hat{w}_{MAP}^T) - \E(\hat{w}_{MAP})\E(\hat{w}_{MAP})^T\\
&= \Lambda_N \mathscr{X}^T\mathscr{X}\Lambda_N\\
&= (\mathscr{X}^T\mathscr{X} + \lambda_0^{-1}I)^{-1}\mathscr{X}^T \mathscr{X}(\mathscr{X}^T\mathscr{X} + \lambda_0^{-1}I)^{-1}. \tag 4 \label{eq-4}
\end{align*}
$$

It is perhaps difficult to compare \eqref{eq-3} and \eqref{eq-4}. But if we take a look at one-dimensional case, it becomes easier to see that, as long as $\lambda_0 >1$,

$$
\Var(\hat{w}_{ML})={1\over \sum_{i=1}^N x_i^2} > {\lambda_0\sum_{i=1}^N x_i^2\over (1+\lambda_0\sum_{i=1}^N x_i^2)^2 } = \Var(\hat{w}_{MAP}).
$$

From the above analysis, we can see that the MAP estimate reduces the variance at the expense of increasing the bias. However, the goal is to prevent overfitting.

