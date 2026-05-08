This note gives the expressions of the gradient and Hessian of the objective function

$$
f(x,y) = \frac{P_0}{M}\frac{y}{x} + \gamma\frac{y}{x}\left(\frac{1}{y(y-K)}\sum_{k=1}^K\frac{\sigma_k^2}{\beta_k}
\left(2^{R_kx}-1\right)\right)^\alpha + \frac{P_1}{M}y
$$

where the parameters are $K>0$, $M\geq K$, $P_0\geq0$, $P_1\geq0$, $P_\mathrm{max}\geq0$, $\gamma\geq0$, $\alpha\in[0.5,1]$, 
$\sigma_k^2\geq0,\ \beta_k\geq0,\ R_k\geq0\ \forall k=1,\dotsc,K$.

Let us define the functions

$$
\phi(x) = \sum_{k=1}^K\frac{\sigma_k^2}{\beta_k}\left(2^{R_kx}-1\right), \quad 
\phi'(x) = \sum_{k=1}^K\frac{\sigma_k^2}{\beta_k}R_k2^{R_kx}, \quad 
\phi''(x) = \sum_{k=1}^K\frac{\sigma_k^2}{\beta_k}R_k^22^{R_kx}
$$

so that we can rewrite

$$
f(x,y) = \frac{P_0}{M}\frac{y}{x} + \gamma\frac{y}{x}\left(\frac{1}{y(y-K)}\phi(x)\right)^\alpha + \frac{P_1}{M}y, \quad 
\mathcal{D} = \big\lbrace x\geq1,\ y\leq M,\ y\geq \frac{K}{2}+\frac{1}{2}
\sqrt{K^2+\frac{4}{P_\mathrm{max}}\phi(x)} \big\rbrace.
$$

We can now express the gradient of $f(x,y)$ as

$$
\nabla f(x,y) = \\
\begin{bmatrix}
-\displaystyle\frac{yP_0}{x^2M} + \frac{y\gamma}{\left(y(y-K)\right)^\alpha}\frac{\alpha\phi(x)^{\alpha-1}
\phi'(x)x-\phi(x)^\alpha}{x^2} \\
\displaystyle\frac{P_0}{xM} + \frac{\gamma}{x}\phi(x)^{\alpha}
\frac{1}{y^{\alpha}(y-K)^{1+\alpha}}\left(y(1-2\alpha)-K(1-\alpha)\right) + \frac{P_1}{M}
\end{bmatrix}
$$

and the entries of the Hessian matrix of $f(x,y)$ as

$$
\frac{\partial^2 f}{\partial x^2} = 2\frac{yP_0}{x^3M} + 
\frac{y\gamma}{\left(y(y-K)\right)^\alpha}\frac{1}{x^3}\bigg\lbrace x\left[
\alpha(\alpha-1)\phi(x)^{\alpha-2}\phi'(x)^2x +
\alpha\phi(x)^{\alpha-1}
\phi''(x)x\right] 
-2\left[\alpha\phi(x)^{\alpha-1}
\phi'(x)x-\phi(x)^\alpha\right]\bigg\rbrace
$$

$$
\frac{\partial^2 f}{\partial x\partial y} = 
\frac{\partial^2 f}{\partial y\partial x} = 
-\frac{P_0}{x^2M}+\gamma\frac{1}{y^{\alpha}(y-K)^{1+\alpha}}
\left[y(1-2\alpha)-K(1-\alpha)\right] 
\frac{\alpha\phi(x)^{\alpha-1}\phi'(x)x-\phi(x)^\alpha}{x^2}
$$

$$
\frac{\partial^2 f}{\partial y^2} = \frac{\gamma}{x}\phi(x)^{\alpha}
\frac{\alpha}{y^{1+\alpha}(y-K)^{2+\alpha}} \left[-K^2(1-\alpha)+
4Ky(1-\alpha)+2y^2(2\alpha-1)\right].
$$
