# Equation-to-Code Map

Reference: *Fast stochastic subspace identification of densely instrumented bridges using randomized SVD* — Elisa Tomassini, Enrique García-Macías, Filippo Ubertini. Department of Civil and Environmental Engineering, University of Perugia, Via G. Duranti, 93 - 06125 Perugia, Italy.

## Covariance-driven SSI (CoV-SSI)

### Continuous-time state-space model
**Equations**

$$
\dot{\mathbf{x}}(t)=\mathbf{A}_C\mathbf{x}(t)+\mathbf{B}_C\mathbf{u}(t)
$$

$$
\mathbf{y}(t)=\mathbf{C}_C\mathbf{x}(t)+\mathbf{D}_C\mathbf{u}(t)
$$

**Implementation**
- Conceptual only (not explicitly coded). These serve as the theoretical model for later discrete-time steps.

### Eigen-decomposition of the continuous state matrix
**Equation**

$$
\mathbf{A}_C=\boldsymbol{\Psi}\boldsymbol{\Lambda}_C\boldsymbol{\Psi}^{-1}
$$

**Implementation**
- Conceptual only. The code performs eigen-decomposition on the discrete-time \(\mathbf{A}\) (see “Discrete eigen-decomposition” below).

### Modal parameters from eigenvalues
**Equation**

$$
\lambda_i=-\xi_i\,\omega_i+\mathrm{i}\,\omega_i\sqrt{1-\xi_i^2}
$$

**Implementation**
- Used implicitly via standard SSI definitions of damping and frequency from continuous poles in `modalID`.
- Implemented in:
  - `fast_ssi/SSICOV.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.modalID`

### Observable modal matrix
**Equation**

$$
\boldsymbol{\Phi}=\mathbf{C}_C\boldsymbol{\Psi}
$$

**Implementation**
- Implemented as `phi0 = C @ eigvecs` inside `modalID`.
- Files:
  - `fast_ssi/SSICOV.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.modalID`

### Discrete-time stochastic state-space model
**Equations**

$$
\mathbf{x}_{h+1}=\mathbf{A}\mathbf{x}_h+\mathbf{w}_h
$$

$$
\mathbf{y}_h=\mathbf{C}\mathbf{x}_h+\mathbf{v}_h
$$

**Implementation**
- Implemented implicitly through the identification of \(\mathbf{A}\) and \(\mathbf{C}\) in `modalID`.
- Files:
  - `fast_ssi/SSICOV.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.modalID`

### Output correlation Toeplitz matrix
**Equation**

$$
\mathbf{T}_{1\mid j_b}=
\begin{bmatrix}
\mathbf{R}_{j_b} & \mathbf{R}_{j_b-1} & \cdots & \mathbf{R}_1 \\
\mathbf{R}_{j_b+1} & \mathbf{R}_{j_b} & \cdots & \mathbf{R}_2 \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{R}_{2j_b-1} & \mathbf{R}_{2j_b-2} & \cdots & \mathbf{R}_{j_b}
\end{bmatrix}
$$

**Implementation**
- Toeplitz construction:
  - `fast_ssi/SSICOV_RSVD.py` → `build_block_toeplitz`
  - `fast_ssi/SSICOV.py` → `blockToeplitz_jit`
  - `fast_ssi/SSICOV_GPU.py` → `blockToeplitz_jit`

### Toeplitz factorization into observability/controllability
**Equation**

$$
\mathbf{T}_{1\mid j_b}=
\begin{bmatrix}
\mathbf{C} \\
\mathbf{C A} \\
\vdots \\
\mathbf{C A}^{j_b-1}
\end{bmatrix}
\begin{bmatrix}
\mathbf{A}^{j_b-1}\mathbf{G} & \cdots & \mathbf{A}\mathbf{G} & \mathbf{G}
\end{bmatrix}
=\mathbf{O}\,\boldsymbol{\Gamma}
$$

**Implementation**
- Implicit: SVD/RSVD produces `U, S, V` and then observability is formed from `U` and `S`.
- Files:
  - `fast_ssi/SSICOV.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.modalID`

### SVD of Toeplitz matrix
**Equation**

$$
\mathbf{T}_{1\mid j_b}=\mathbf{U}\,\mathbf{S}\,\mathbf{V}^T
$$

**Implementation**
- Standard SVD:
  - `fast_ssi/SSICOV.py` → `blockToeplitz_jit`
  - `fast_ssi/SSICOV_GPU.py` → `blockToeplitz_jit`
- Randomized SVD (RSVD):
  - `fast_ssi/SSICOV_RSVD.py` → `randomized_svd`

### Observability/controllability from SVD
**Equations**

$$
\mathbf{O}=\mathbf{U}\,\mathbf{S}^{1/2}
$$

$$
\boldsymbol{\Gamma}=\mathbf{S}^{1/2}\,\mathbf{V}^T
$$

**Implementation**
- `obs = U[:, :Nmodes] @ sqrt(S)`
- Files:
  - `fast_ssi/SSICOV.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.modalID`

### State matrix estimation from observability blocks
**Equation**

$$
\mathbf{A}=\mathbf{O}^{to\,\dagger}\,\mathbf{O}^{bo}
$$

**Implementation**
- `A = pinv(obs[0:ao, :]) @ obs[bo:co, :]`
- Files:
  - `fast_ssi/SSICOV.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.modalID`

### Discrete eigen-decomposition of A
**Equation**

$$
\mathbf{A}=\boldsymbol{\Psi}\,\mathbf{M}\,\boldsymbol{\Psi}^{-1}
$$

**Implementation**
- `eigvals, eigvecs = np.linalg.eig(A)`
- Files:
  - `fast_ssi/SSICOV.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.modalID`

### Continuous poles, frequencies, damping (standard SSI convention used in code)
**Equations**

$$
\lambda=\frac{\ln(\mu)}{\Delta t}
$$

$$
f_n=\frac{|\lambda|}{2\pi}
$$

$$
\zeta=-\frac{\mathrm{Re}(\lambda)}{|\lambda|}
$$

**Implementation**
- Implemented in `modalID` for all three variants.
- Conjugate-pair selection uses `Im(\lambda) > 0` (one pole per pair).
- Files:
  - `fast_ssi/SSICOV.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.modalID`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.modalID`

## Randomized SVD (RSVD)

### Low-rank approximation
**Equation**

$$
\mathbf{A}\approx\mathbf{Q}\,\mathbf{Q}^T\,\mathbf{A}
$$

**Implementation**
- `Y = A @ Ω` followed by QR to build `Q`.
- File: `fast_ssi/SSICOV_RSVD.py` → `randomized_svd`

### Random projection
**Equation**

$$
\mathbf{Y}=\mathbf{A}\,\boldsymbol{\Omega}
$$

**Implementation**
- `Y = T1 @ random_matrix`
- File: `fast_ssi/SSICOV_RSVD.py` → `randomized_svd`

### Projected matrix and SVD
**Equations**

$$
\mathbf{P}=\mathbf{Q}^T\,\mathbf{A}
$$

$$
\mathbf{P}=\tilde{\mathbf{U}}\,\boldsymbol{\Sigma}\,\mathbf{V}^T
$$

**Implementation**
- `B = Q.conj().T @ T1` then `svd(B)`
- File: `fast_ssi/SSICOV_RSVD.py` → `randomized_svd`

### Reconstruct RSVD
**Equation**

$$
\mathbf{A}\approx\mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^T,\quad \mathbf{U}=\mathbf{Q}\,\tilde{\mathbf{U}}
$$

**Implementation**
- `U = Q @ U_tilde`
- File: `fast_ssi/SSICOV_RSVD.py` → `randomized_svd`

## Stabilization and pole filtering

### Stability checks (frequency, damping, MAC)
**Equation (as described in markdown text)**

$$
\frac{|f_i^m-f_j^{m-1}|}{\max(f_i^m,f_j^{m-1})} \le \alpha_f
$$

$$
\frac{|\xi_i^m-\xi_j^{m-1}|}{\max(\xi_i^m,\xi_j^{m-1})} \le \alpha_\xi
$$

$$
\mathrm{MAC}(\boldsymbol{\phi}_i^m,\boldsymbol{\phi}_j^{m-1}) \ge \alpha_{\mathrm{MAC}}
$$

**Implementation**
- `stabilityCheck` compares consecutive model orders using `errorcheck` and `getMAC`.
- Files:
  - `fast_ssi/SSICOV.py` → `SSICOV.stabilityCheck`
  - `fast_ssi/SSICOV_RSVD.py` → `SSICOV.stabilityCheck`
  - `fast_ssi/SSICOV_GPU.py` → `SSICOV.stabilityCheck`

### 3D stabilization across time-lag
**Equation (3D stability condition with time-lag axis)**

$$
\begin{aligned}
\frac{|f_i^m-f_j^{m-1}|}{\max(f_i^m,f_j^{m-1})} &\vee \frac{|f_i^t-f_j^{t-1}|}{\max(f_i^t,f_j^{t-1})} \\
\frac{|\xi_i^m-\xi_j^{m-1}|}{\max(\xi_i^m,\xi_j^{m-1})} &\vee \frac{|\xi_i^t-\xi_j^{t-1}|}{\max(\xi_i^t,\xi_j^{t-1})} \\
\mathrm{MAC}(\boldsymbol{\phi}_i^m,\boldsymbol{\phi}_j^{m-1}) &\vee \mathrm{MAC}(\boldsymbol{\phi}_i^t,\boldsymbol{\phi}_j^{t-1}) 
\end{aligned}
$$

**Implementation**
- Not implemented. Current code only compares adjacent model orders at a fixed time-lag.

## Notes on pairing and complex mode shapes

- Conjugate-pair selection is done by retaining only poles with `Im(\lambda) > 0`.
- Mode shapes are kept complex; a phase rotation is applied for consistent sign using the first DOF as reference.
- These behaviors are implemented in `modalID` and `getStablePoles` for all three variants:
  - `fast_ssi/SSICOV.py`
  - `fast_ssi/SSICOV_RSVD.py`
  - `fast_ssi/SSICOV_GPU.py`
