# Methodology

This guide describes the diagnostic in *Distributional Pre-Trends: A Variance Diagnostic for Difference-in-Differences Designs* by Aditya Alaparti, Amann Katta, Nithin Vargese, and Swaraj Sainani (12 October 2026). See [attribution](attribution.md). Mathematical statements concern the stated sampling conditions; numerical validity flags do not establish them.

## Population target

Let group membership be time-invariant, $D_i\in\{0,1\}$, and let $K\geq3$ common periods precede treatment. For the complete outcome trajectory $Y_i$, define

$$
M_K=I_K-\frac{1}{K}\mathbf1\mathbf1',\qquad
Z_i=M_KY_i,\qquad q_{gt}=\operatorname{Var}(Z_{it}\mid D_i=g).
$$

The variance null is

$$
H_{0,q}:\quad \rho_t=\log(q_{1t}/q_{0t})=\rho_1,
\quad t=2,\ldots,K,
$$

with all $q_{gt}>0$. It permits a constant ratio other than one. The outcome scale, sample, and common demeaning window define this target and should be chosen before looking at test results.

If $Y_{it}(0)=\alpha_i+\tau_t+\varepsilon_{it}$, demeaning removes the unit constant, but it also transforms the disturbance covariance:

$$
q_{gt}=[M_K\Omega_gM_K]_{tt},\qquad
\Omega_g=\operatorname{Var}(\varepsilon_i\mid g).
$$

Consequently, the target generally differs from the raw-outcome variance and the original disturbance variance. Different serial dependence can produce different demeaned ratios even when raw marginal variance ratios are constant. Under serially uncorrelated disturbances and $K>2$, proportionality of the original disturbance variance paths is equivalent to proportionality of the transformed paths. That equivalence does not extend automatically to raw outcomes or arbitrary serial dependence.

With two periods, $Z_{i2}=-Z_{i1}$, so both within-group sample variances are identical and the log-ratio contrast is exactly zero. The public interface requires at least three periods. Three periods alone do not ensure a nonsingular covariance.

## Estimation and full temporal covariance

For $N_g$ units in group $g$, compute group-centered residuals $\widehat X_{git}=Z_{it}-\bar Z_{gt}$ and sample variances

$$
s^2_{gt}=\frac{1}{N_g-1}\sum_i\widehat X_{git}^{,2},\qquad
\widehat\rho_t=\log s^2_{1t}-\log s^2_{0t}.
$$

Let each row of $A$ contrast a nonreference period with the reference period. The variance restriction estimate is $\widehat d=A\widehat\rho$. The returned ratios are $s^2_{1t}/s^2_{0t}$, not raw-outcome variance ratios.

The population log-variance influence vector has coordinates

$$
\psi_{git}=X_{git}^{2}/q_{gt}-1,\qquad
C_g=E[\psi_{gi}\psi_{gi}'].
$$

For independent units and independent groups,

$$
V_\rho=C_1/N_1+C_0/N_0,\qquad V_d=AV_\rho A'.
$$

The implementation uses $\widetilde q_{gt}=N_g^{-1}\sum_i\widehat X_{git}^{,2}$ in the estimated influence vectors and $N_g^{-1}\sum_i\widehat\psi_{gi}\widehat\psi_{gi}'$ for $C_g$. The $N_g-1$ versus $N_g$ adjustment in the point variances contributes a period-constant term to the log ratio and cancels from its contrasts.

The full matrix includes covariance between squared residuals at different periods. Its diagonal contains the marginal kurtosis corrections, but those corrections alone are insufficient. Demeaning induces temporal dependence even when the original disturbances are serially independent. Under serially iid Gaussian disturbances of constant variance, the full contrast covariance includes the factor $1-(K-1)^{-2}$; omitting it gives an incorrect chi-square scaling.

The variance Wald statistic is

$$
W_q=\widehat d'\widehat V_d^{-1}\widehat d.
$$

Under its null and the [regularity conditions](assumptions.md), its limiting distribution is $\chi^2_{K-1}$. This is an asymptotic statement. A small numerical p-value is not an exact finite-sample probability. The implementation requires the smallest covariance eigenvalue to exceed $10^{-10}$ times the largest and does not silently substitute a pseudoinverse or reduced degrees of freedom.

## Mean diagnostic

The mean restriction estimate is

$$
\widehat b=A(\bar Z_1-\bar Z_0).
$$

Since $AM_K=A$, it also equals the contrast of raw between-group mean differences. Its covariance uses the unbiased trajectory covariance in each group divided by that group's size, contrasted by $A$. The mean Wald statistic $W_m=\widehat b'\widehat V_b^{-1}\widehat b$ has an asymptotic $\chi^2_{K-1}$ reference under its null. This is not the special classical mean F test with a scalar pooled residual variance.

## Clusters and combined diagnostics

For clusters, sum signed unit influence vectors within the independent cluster before taking outer products. Treated units receive weight $1/N_1$ and controls $-1/N_0$. A cluster shared across groups therefore retains cross-group covariance. Group-specific equal-size strata use the group multiplier $G_g/(G_g-1)$; shared-cluster inference uses the pooled multiplier $G/(G-1)$. These are finite-cluster conventions, not exact small-cluster calibrations.

The optional Fisher statistic is $-2(\log p_m+\log p_q)$ with a $\chi^2_4$ reference. It requires valid, jointly asymptotically independent component p-values under the joint null. It is disabled by default and requires an explicit justification when enabled. Whole-trajectory central symmetry with consistent covariance estimation is a sufficient independent-unit condition; marginal symmetry alone is not.

The other combination is $T=W_m+W_q$, calibrated by the [centered bootstrap](bootstrap.md). Both roots use the same resampled trajectories. This preserves their dependence without constructing or inverting a full joint covariance matrix. No general chi-square law is assigned to this sum.

## Interpretation

Equal CDF changes across groups do not imply a constant central-variance ratio. Mean parallel trends and a constant variance ratio are also distinct restrictions. Rejection of this variance test therefore does not by itself establish failure of a CDF restriction, failure of mean parallel trends, or bias in an average treatment effect.

The variance contrast is invariant to fixed nonzero rescaling within each group, unit-specific additive constants, and common additive time effects. Nonlinear transformations can change the target and decision. Changes to the demeaning window can also change the target.

Combining diagnostics has no uniform power advantage. Inspecting several p-values and declaring a design problematic whenever any crosses $\alpha$ does not control the probability of any rejection at $\alpha$. Prespecify the primary diagnostic and any multiplicity rule. Nonrejection can reflect low power; it is not an equivalence test.
