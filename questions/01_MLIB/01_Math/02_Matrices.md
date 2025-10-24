# Matrices

Question 1. [E] Why do we say that matrices are linear transformations?

**Solution:** Matrices represent linear transformations because they satisfy the fundamental properties of linearity:

**Intuitive Understanding:**
Think of a matrix as a "machine" that takes vectors as input and produces vectors as output. This machine is linear if it preserves two key properties: scaling and addition.

**Mathematical Definition:**
A transformation $T$ is linear if for any vectors $u, v$ and scalar $c$:
1. **Additivity**: $T(u + v) = T(u) + T(v)$
2. **Homogeneity**: $T(cu) = cT(u)$

**Why Matrices Are Linear Transformations:**

**Step 1: Matrix-vector multiplication**
- For matrix $A$ and vector $x$, the transformation is $T(x) = Ax$
- This is the fundamental operation that matrices perform

**Step 2: Verify additivity**
- $T(u + v) = A(u + v) = Au + Av = T(u) + T(v)$ ✓
- Matrix multiplication distributes over vector addition

**Step 3: Verify homogeneity**
- $T(cu) = A(cu) = c(Au) = cT(u)$ ✓
- Scalar multiplication can be factored out

**ML Applications:**

**1. Neural Network Layers:**
- **Linear layers**: $y = Wx + b$ where $W$ is a weight matrix
- **Convolution**: Convolutional layers apply linear transformations to image patches
- **Attention**: Attention mechanisms use linear transformations to compute query, key, value

**2. Data Preprocessing:**
- **PCA**: Principal Component Analysis uses linear transformations to reduce dimensionality
- **Whitening**: Data normalization involves linear transformations
- **Feature scaling**: Linear transformations standardize features

**3. Optimization:**
- **Gradient descent**: Updates involve linear combinations of gradients
- **Momentum**: Linear combinations of past gradients
- **Adam**: Adaptive learning rates use linear transformations

**Geometric Interpretation:**
- **Rotation**: Rotation matrices preserve angles and distances
- **Scaling**: Diagonal matrices scale along coordinate axes
- **Shearing**: Non-diagonal matrices can skew shapes
- **Reflection**: Matrices with determinant -1 flip orientation

**Key Insight**: Every matrix operation in ML (forward pass, backpropagation, optimization) relies on the linearity property. This is why we can use efficient matrix operations and why gradient-based optimization works so well.

Question 2. [E] What's the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?

**Solution:** Matrix inverses are fundamental in ML for solving linear systems and understanding model behavior:

**Intuitive Understanding:**
Think of a matrix inverse as the "undo" operation. If a matrix $A$ transforms a vector $x$ to $y = Ax$, then the inverse $A^{-1}$ transforms $y$ back to $x = A^{-1}y$.

**Mathematical Definition:**
For a square matrix $A$, the inverse $A^{-1}$ satisfies:
- $AA^{-1} = A^{-1}A = I$ (identity matrix)
- $A^{-1}A = I$ means "applying $A$ then $A^{-1}$ returns the original vector"

**Do All Matrices Have an Inverse?**

**Answer: No** - Only square matrices with non-zero determinant have inverses.

**Conditions for invertibility:**
1. **Square matrix**: Must be $n \times n$
2. **Non-singular**: $\det(A) \neq 0$
3. **Full rank**: $\text{rank}(A) = n$
4. **Linearly independent columns/rows**

**Step-by-Step Examples:**

**Example 1: Invertible Matrix**
- $`A = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}`$
- $\det(A) = 2(1) - 1(1) = 1 \neq 0$ ✓
- $`A^{-1} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}`$
- Verification: $`AA^{-1} = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`$ ✓

**Example 2: Non-invertible Matrix**
- $`A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}`$
- $\det(A) = 1(4) - 2(2) = 0$ ✗
- Second row is 2× first row (linearly dependent)
- No inverse exists

**Is the Inverse Always Unique?**

**Answer: Yes** - If an inverse exists, it is unique.

**Proof by contradiction:**
- Suppose $B$ and $C$ are both inverses of $A$
- Then $B = BI = B(AC) = (BA)C = IC = C$
- Therefore, $B = C$ (unique)

**ML Applications:**

**1. Solving Linear Systems:**
- **Problem**: $Ax = b$ where $A$ is invertible
- **Solution**: $x = A^{-1}b$
- **Example**: Solving normal equations in linear regression

**2. Neural Network Training:**
- **Weight updates**: Some optimization methods use matrix inverses
- **Hessian matrix**: Second-order optimization requires inverse of Hessian
- **Batch normalization**: Inverting covariance matrices

**3. Data Analysis:**
- **Covariance matrix**: Inverting for Mahalanobis distance
- **Precision matrix**: Inverse of covariance matrix in Gaussian models
- **Whitening**: $x_{whitened} = \Sigma^{-1/2}(x - \mu)$

**4. Regularization:**
- **Ridge regression**: $(X^TX + \lambda I)^{-1}X^Ty$
- **L2 regularization**: Prevents singular matrices
- **Numerical stability**: Adding small values to diagonal

**Computational Considerations:**

**When to use matrix inverse:**
- **Small matrices**: Direct computation is feasible
- **One-time computation**: Inverse computed once, used many times
- **Theoretical analysis**: Understanding model behavior

**When to avoid matrix inverse:**
- **Large matrices**: Computationally expensive $O(n^3)$
- **Numerical stability**: Near-singular matrices cause errors
- **Iterative methods**: Use conjugate gradient, Cholesky decomposition

**Key Insight**: In ML, we often avoid computing inverses directly due to computational cost and numerical issues. Instead, we use iterative methods, factorizations, or regularization techniques.

Question 3. [E] What does the determinant of a matrix represent?

**Solution:** The determinant is a fundamental scalar value that captures important geometric and algebraic properties of matrices:

**Intuitive Understanding:**
Think of the determinant as a "scaling factor" that tells you how much a matrix transformation changes the volume (or area in 2D) of shapes. It also indicates whether the transformation preserves or flips orientation.

**Geometric Interpretation:**

**2D Case:**
- **Area scaling**: $\det(A)$ = factor by which area is scaled
- **Orientation**: 
  - $\det(A) > 0$: preserves orientation (no flipping)
  - $\det(A) < 0$: flips orientation (mirror reflection)
  - $\det(A) = 0$: collapses to lower dimension (singular)

**3D Case:**
- **Volume scaling**: $\det(A)$ = factor by which volume is scaled
- **Orientation**: Same sign rules as 2D

**Mathematical Properties:**

**1. Multiplicative Property:**
- $\det(AB) = \det(A)\det(B)$
- **ML Insight**: Composing transformations multiplies their effects

**2. Transpose Property:**
- $\det(A^T) = \det(A)$
- **ML Insight**: Row and column operations have same effect

**3. Inverse Property:**
- $\det(A^{-1}) = \frac{1}{\det(A)}$ (if $A$ is invertible)
- **ML Insight**: Inverse transformation has reciprocal scaling

**Step-by-Step Examples:**

**Example 1: 2×2 Matrix**
- $`A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}`$
- $\det(A) = 3(2) - 1(0) = 6$
- **Interpretation**: Area is scaled by factor of 6, orientation preserved

**Example 2: Rotation Matrix**
- $`R = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}`$
- $\det(R) = \cos^2\theta + \sin^2\theta = 1$
- **Interpretation**: Rotation preserves area and orientation

**Example 3: Singular Matrix**
- $`A = \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix}`$
- $\det(A) = 2(2) - 4(1) = 0$
- **Interpretation**: Collapses 2D space to 1D line

**ML Applications:**

**1. Matrix Invertibility:**
- **Condition**: $A$ is invertible if and only if $\det(A) \neq 0$
- **Use**: Quick check for singularity before computing inverse
- **Example**: Checking if weight matrices are invertible

**2. Volume/Area Calculations:**
- **Covariance matrices**: $\det(\Sigma)$ measures spread of multivariate data
- **Jacobian**: $\det(J)$ in change of variables for probability distributions
- **Data preprocessing**: Understanding how transformations affect data spread

**3. Optimization:**
- **Hessian matrix**: $\det(H)$ indicates curvature of loss landscape
- **Positive definite**: All eigenvalues > 0 implies $\det(H) > 0$
- **Saddle points**: $\det(H) = 0$ indicates potential saddle points

**4. Neural Networks:**
- **Weight initialization**: Avoid singular matrices (det = 0)
- **Gradient flow**: Determinant affects how gradients propagate
- **Batch normalization**: Preserves determinant of transformations

**5. Dimensionality Reduction:**
- **PCA**: Determinant of covariance matrix measures total variance
- **LDA**: Determinant ratio measures class separability
- **Feature selection**: Remove features that make determinant too small

**Computational Considerations:**

**Numerical Stability:**
- **Small determinants**: Near-singular matrices cause numerical issues
- **Condition number**: $\kappa(A) = \frac{\sigma_{max}}{\sigma_{min}}$ relates to determinant
- **Regularization**: Add small values to diagonal to avoid singular matrices

**Efficient Computation:**
- **LU decomposition**: $\det(A) = \det(L)\det(U) = \prod_i L_{ii}U_{ii}$
- **Cholesky decomposition**: For positive definite matrices
- **SVD**: $\det(A) = \prod_i \sigma_i$ (product of singular values)

**Key Insight**: The determinant is crucial in ML for understanding model behavior, ensuring numerical stability, and analyzing the geometry of transformations. It's particularly important in optimization, where singular matrices can cause training instability.

Question 4. [E] What happens to the determinant of a matrix if we multiply one of its rows by a scalar $t \in \mathbb{R}$?

**Solution:** Multiplying a row by a scalar $t$ multiplies the determinant by $t$:

**Mathematical Statement:**
If $B$ is obtained from $A$ by multiplying one row by $t$, then $\det(B) = t \cdot \det(A)$

**Intuitive Understanding:**
Think of the determinant as measuring volume. If you stretch one dimension by a factor of $t$, the volume changes by that same factor.

**Step-by-Step Proof:**

**Step 1: Start with original matrix**
- $`A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{bmatrix}`$

**Step 2: Multiply first row by $t$**
- $`B = \begin{bmatrix} ta_{11} & ta_{12} & \cdots & ta_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{bmatrix}`$

**Step 3: Use determinant properties**
- $\det(B) = t \cdot \det(A)$ (by linearity in rows)

**Concrete Example:**

**Original matrix:**
- $`A = \begin{bmatrix} 2 & 3 \\ 1 & 4 \end{bmatrix}`$
- $\det(A) = 2(4) - 3(1) = 8 - 3 = 5$

**Multiply first row by $t = 3$:**
- $`B = \begin{bmatrix} 6 & 9 \\ 1 & 4 \end{bmatrix}`$
- $\det(B) = 6(4) - 9(1) = 24 - 9 = 15$
- **Verification**: $15 = 3 \times 5 = t \times \det(A)$ ✓

**ML Applications:**

**1. Data Preprocessing:**
- **Feature scaling**: Multiplying features by constants affects covariance matrix determinant
- **Normalization**: Understanding how scaling affects data spread
- **Whitening**: ZCA whitening preserves determinant relationships

**2. Neural Network Training:**
- **Weight initialization**: Scaling weights affects gradient magnitudes
- **Learning rate**: Understanding how parameter scaling affects optimization
- **Batch normalization**: Preserves determinant of transformations

**3. Optimization:**
- **Hessian scaling**: Scaling parameters affects condition number
- **Gradient descent**: Understanding how scaling affects convergence
- **Adaptive methods**: Adam, RMSprop adjust for parameter scales

**4. Regularization:**
- **L2 regularization**: Adding $\lambda I$ to diagonal affects determinant
- **Ridge regression**: $(X^TX + \lambda I)^{-1}$ determinant changes with $\lambda$
- **Numerical stability**: Small $\lambda$ prevents singular matrices

**General Properties:**

**Multiple row scaling:**
- If $k$ rows are multiplied by $t_1, t_2, \ldots, t_k$
- Then $\det(B) = t_1 t_2 \cdots t_k \cdot \det(A)$

**Column scaling:**
- Same property holds for columns: $\det(A^T) = \det(A)$

**Zero scaling:**
- If $t = 0$, then $\det(B) = 0$ (singular matrix)
- **ML Insight**: Zeroing out features makes matrix singular

**Negative scaling:**
- If $t < 0$, determinant changes sign
- **ML Insight**: Flipping feature signs affects orientation

**Key Insight**: This property is fundamental in ML for understanding how data preprocessing, feature scaling, and parameter initialization affect model behavior. It's crucial for maintaining numerical stability and understanding the geometry of transformations.

Question 5. [M] A $4 \times 4$ matrix has four eigenvalues $3, 3, 2, -1$. What can we say about the trace and the determinant of this matrix?

**Solution:** We can determine both the trace and determinant using the eigenvalues:

**Intuitive Understanding:**
Eigenvalues are the "fundamental frequencies" of a matrix transformation. The trace and determinant are simple functions of these eigenvalues that capture important properties of the transformation.

**Mathematical Relationships:**

**1. Trace from Eigenvalues:**
- $\text{tr}(A) = \sum_{i=1}^n \lambda_i$ (sum of all eigenvalues)
- **For our matrix**: $\text{tr}(A) = 3 + 3 + 2 + (-1) = 7$

**2. Determinant from Eigenvalues:**
- $\det(A) = \prod_{i=1}^n \lambda_i$ (product of all eigenvalues)
- **For our matrix**: $\det(A) = 3 \times 3 \times 2 \times (-1) = -18$

**Step-by-Step Calculation:**

**Given eigenvalues**: $\lambda_1 = 3, \lambda_2 = 3, \lambda_3 = 2, \lambda_4 = -1$

**Trace calculation:**
- $\text{tr}(A) = \lambda_1 + \lambda_2 + \lambda_3 + \lambda_4$
- $\text{tr}(A) = 3 + 3 + 2 + (-1) = 7$

**Determinant calculation:**
- $\det(A) = \lambda_1 \times \lambda_2 \times \lambda_3 \times \lambda_4$
- $\det(A) = 3 \times 3 \times 2 \times (-1) = 18 \times (-1) = -18$

**Verification with Characteristic Polynomial:**
- $p(\lambda) = (\lambda - 3)^2(\lambda - 2)(\lambda + 1)$
- $p(\lambda) = \lambda^4 - 7\lambda^3 + 11\lambda^2 + 7\lambda - 18$
- **Trace**: Coefficient of $\lambda^3$ = 7 ✓
- **Determinant**: Constant term = -18 ✓

**ML Applications:**

**1. Matrix Analysis:**
- **Positive definite**: All eigenvalues > 0 (not our case: -1 < 0)
- **Positive semi-definite**: All eigenvalues ≥ 0 (not our case)
- **Invertible**: No zero eigenvalues (✓ our case: all non-zero)

**2. Optimization:**
- **Hessian matrix**: Eigenvalues indicate curvature
- **Saddle points**: Mixed positive/negative eigenvalues (our case: 3,3,2 > 0, -1 < 0)
- **Convergence**: Negative eigenvalues can cause instability

**3. Principal Component Analysis:**
- **Covariance matrix**: Eigenvalues = variances along principal components
- **Data spread**: Large eigenvalues indicate high variance directions
- **Dimensionality**: Number of significant eigenvalues

**4. Neural Networks:**
- **Weight matrices**: Eigenvalues affect gradient flow
- **Vanishing gradients**: Small eigenvalues cause slow learning
- **Exploding gradients**: Large eigenvalues cause instability

**5. Regularization:**
- **Ridge regression**: Adding $\lambda I$ shifts eigenvalues by $\lambda$
- **L2 regularization**: $(A + \lambda I)$ has eigenvalues $\lambda_i + \lambda$
- **Numerical stability**: Prevents near-zero eigenvalues

**Geometric Interpretation:**

**Eigenvalue 3 (multiplicity 2):**
- **Stretching**: Vectors in this eigenspace are stretched by factor 3
- **Stability**: Positive eigenvalue indicates stable direction

**Eigenvalue 2:**
- **Stretching**: Vectors in this eigenspace are stretched by factor 2
- **Stability**: Positive eigenvalue indicates stable direction

**Eigenvalue -1:**
- **Reflection**: Vectors in this eigenspace are reflected
- **Instability**: Negative eigenvalue indicates unstable direction

**Key Insights:**

**1. Trace = 7:**
- Sum of diagonal elements
- Measures "total scaling" in all directions
- Used in matrix norms and regularization

**2. Determinant = -18:**
- Negative determinant indicates orientation flip
- Volume scaling by factor 18
- Sign change due to negative eigenvalue

**3. Mixed Eigenvalues:**
- Indicates saddle point in optimization
- Some directions stable (positive), some unstable (negative)
- Common in non-convex optimization landscapes

**Key Insight**: The eigenvalues reveal that this matrix represents a transformation with both stable and unstable directions, which is common in ML optimization problems. The negative determinant indicates an orientation flip, and the mixed signs suggest potential optimization challenges.

Question 6. [M] Given the following matrix:
   
$$
\begin{bmatrix}
1 & 4 & -2 \\
-1 & 3 & 2 \\
3 & 5 & -6
\end{bmatrix}
$$

   Without explicitly using the equation for calculating determinants, what can we say about this matrix's determinant?

   **Hint:** Rely on a property of this matrix to determine its determinant.

**Solution:** We can determine the determinant by analyzing the linear dependence of the rows:

**Step 1: Analyze Row Relationships**
- Row 1: $[1, 4, -2]$
- Row 2: $[-1, 3, 2]$
- Row 3: $[3, 5, -6]$

**Step 2: Check for Linear Dependence**
Let's see if any row can be expressed as a linear combination of others:

**Check if Row 3 = a×Row 1 + b×Row 2:**
- $3 = a(1) + b(-1) = a - b$ ... (1)
- $5 = a(4) + b(3) = 4a + 3b$ ... (2)
- $-6 = a(-2) + b(2) = -2a + 2b$ ... (3)

**Solving equations (1) and (2):**
- From (1): $a = 3 + b$
- Substituting into (2): $5 = 4(3 + b) + 3b = 12 + 4b + 3b = 12 + 7b$
- Therefore: $7b = -7$, so $b = -1$
- Therefore: $a = 3 + (-1) = 2$

**Verification with equation (3):**
- $-6 = -2(2) + 2(-1) = -4 - 2 = -6$ ✓

**Step 3: Conclusion**
Since Row 3 = 2×Row 1 + (-1)×Row 2, the rows are linearly dependent.

**Step 4: Determinant Property**
- **Key property**: If rows of a matrix are linearly dependent, then $\det(A) = 0$
- **Reason**: Linear dependence means the matrix collapses to a lower-dimensional space
- **Geometric interpretation**: The volume of the parallelepiped formed by the row vectors is zero

**ML Applications:**

**1. Feature Analysis:**
- **Redundant features**: Linearly dependent rows indicate redundant features
- **Feature selection**: Remove dependent features to avoid multicollinearity
- **Dimensionality reduction**: Identify the true dimensionality of the feature space

**2. Data Preprocessing:**
- **Singular matrices**: Check for linear dependence before matrix operations
- **Numerical stability**: Avoid near-singular matrices in optimization
- **Regularization**: Add small values to diagonal to prevent singularity

**3. Neural Networks:**
- **Weight matrices**: Linearly dependent weights can cause training issues
- **Gradient flow**: Dependent rows can cause vanishing or exploding gradients
- **Initialization**: Ensure weight matrices have full rank

**4. Optimization:**
- **Hessian matrix**: Singular Hessian indicates optimization problems
- **Convergence**: Linear dependence can cause slow or unstable convergence
- **Regularization**: L2 regularization prevents singular matrices

**Key Insight**: This matrix is singular (determinant = 0) because its rows are linearly dependent. In ML, this indicates redundant information or potential numerical issues that need to be addressed through feature selection or regularization.

Question 7. [M] What's the difference between the covariance matrix $A^T A$ and the Gram matrix $A A^T$?

**Solution:** These matrices capture different relationships and have distinct properties in ML:

**Intuitive Understanding:**
- **$A^T A$ (Covariance-like)**: Measures how features relate to each other
- **$A A^T$ (Gram matrix)**: Measures how samples relate to each other

**Mathematical Properties:**

**Covariance Matrix $A^T A$:**
- **Dimensions**: $m \times m$ (where $A$ is $n \times m$)
- **Interpretation**: Feature-to-feature relationships
- **Symmetric**: $(A^T A)^T = A^T A$
- **Positive semi-definite**: $x^T(A^T A)x \geq 0$ for all $x$

**Gram Matrix $A A^T$:**
- **Dimensions**: $n \times n$ (where $A$ is $n \times m$)
- **Interpretation**: Sample-to-sample relationships
- **Symmetric**: $(A A^T)^T = A A^T$
- **Positive semi-definite**: $x^T(A A^T)x \geq 0$ for all $x$

**Step-by-Step Examples:**

**Given matrix**: $`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}`$ (3 samples, 2 features)

**Covariance Matrix $A^T A$:**
- $`A^T = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix}`$
- $`A^T A = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} = \begin{bmatrix} 35 & 44 \\ 44 & 56 \end{bmatrix}`$
- **Dimensions**: $2 \times 2$ (features × features)
- **Interpretation**: How feature 1 relates to feature 2 across all samples

**Gram Matrix $A A^T$:**
- $`A A^T = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix} = \begin{bmatrix} 5 & 11 & 17 \\ 11 & 25 & 39 \\ 17 & 39 & 61 \end{bmatrix}`$
- **Dimensions**: $3 \times 3$ (samples × samples)
- **Interpretation**: How sample 1 relates to sample 2 across all features

**ML Applications:**

**1. Principal Component Analysis (PCA):**
- **Covariance matrix**: $A^T A$ used to find principal components
- **Eigenvectors**: Directions of maximum variance in feature space
- **Dimensionality reduction**: Project data onto principal components

**2. Kernel Methods:**
- **Gram matrix**: $A A^T$ represents kernel matrix $K(x_i, x_j)$
- **Support Vector Machines**: Use Gram matrix for kernel computations
- **Similarity**: Measures similarity between samples

**3. Neural Networks:**
- **Feature learning**: $A^T A$ captures learned feature relationships
- **Representation**: $A A^T$ captures sample similarities in learned space
- **Attention mechanisms**: Gram matrices used in self-attention

**4. Data Analysis:**
- **Clustering**: Gram matrix used for sample clustering
- **Dimensionality**: Rank of $A^T A$ = rank of $A A^T$ = rank of $A$
- **Condition number**: Both matrices have same condition number

**Computational Considerations:**

**Memory Usage:**
- **$A^T A$**: $m \times m$ - efficient when $m << n$ (few features, many samples)
- **$A A^T$**: $n \times n$ - efficient when $n << m$ (few samples, many features)

**Numerical Stability:**
- **Both matrices**: Can be ill-conditioned for large datasets
- **Regularization**: Add $\lambda I$ to diagonal for stability
- **SVD**: Use SVD for efficient computation

**When to Use Which:**

**Use $A^T A$ when:**
- **Feature analysis**: Understanding feature relationships
- **PCA**: Principal component analysis
- **$m << n$**: Few features, many samples

**Use $A A^T$ when:**
- **Sample analysis**: Understanding sample similarities
- **Kernel methods**: Support vector machines, kernel PCA
- **$n << m$**: Few samples, many features

**Key Insight**: The choice between $A^T A$ and $A A^T$ depends on whether you're analyzing features or samples, and the relative dimensions of your data. Both matrices contain the same information but in different forms, making them suitable for different ML tasks.

Question 8. Given $A \in \mathbb{R}^{n \times m}$ and $b \in \mathbb{R}^n$

   i. [M] Find $x$ such that: $Ax = b$.

   ii. [E] When does this have a unique solution?
   iii. [M] Why is it that when $A$ has more columns than rows, $Ax = b$ has multiple solutions?
   iv. [M] Given a matrix $A$ with no inverse. How would you solve the equation $Ax = b$? What is the pseudoinverse and how to calculate it?

**Solution:**

**i. [M] Find $x$ such that: $Ax = b$.**

**Intuitive Understanding:**
This is the fundamental problem of solving linear systems, which appears everywhere in ML from linear regression to neural network training.

**Mathematical Approach:**

**Case 1: Square matrix ($n = m$) and invertible**
- **Solution**: $x = A^{-1}b$
- **Condition**: $\det(A) \neq 0$ and $\text{rank}(A) = n$

**Case 2: Overdetermined system ($n > m$)**
- **Problem**: More equations than unknowns
- **Solution**: Least squares: $x = (A^T A)^{-1}A^T b$
- **Condition**: $A^T A$ is invertible (columns of $A$ are linearly independent)

**Case 3: Underdetermined system ($n < m$)**
- **Problem**: Fewer equations than unknowns
- **Solution**: Multiple solutions exist
- **Minimum norm solution**: $x = A^T(AA^T)^{-1}b$ (if $AA^T$ is invertible)

**ii. [E] When does this have a unique solution?**

**Answer**: When the system is square and non-singular, or overdetermined with full column rank.

**Conditions for unique solution:**
1. **Square case**: $A$ is $n \times n$ and $\det(A) \neq 0$
2. **Overdetermined case**: $A$ is $n \times m$ with $n > m$ and $\text{rank}(A) = m$
3. **Consistent system**: $b$ is in the column space of $A$

**ML Applications:**
- **Linear regression**: $X^TX$ must be invertible for unique solution
- **Neural networks**: Weight matrices must be full rank
- **Optimization**: Hessian must be positive definite

**iii. [M] Why is it that when $A$ has more columns than rows, $Ax = b$ has multiple solutions?**

**Intuitive Understanding:**
When $A$ has more columns than rows, we have more unknowns than equations, creating "degrees of freedom" that allow multiple solutions.

**Mathematical Explanation:**

**Step 1: Dimension analysis**
- $A$ is $n \times m$ where $n < m$ (more columns than rows)
- Column space of $A$ has dimension $\leq n$
- Null space of $A$ has dimension $\geq m - n > 0$

**Step 2: Solution structure**
- If $x_0$ is any solution to $Ax = b$
- Then $x_0 + v$ is also a solution for any $v$ in null space of $A$
- Since null space has dimension $\geq m - n > 0$, there are infinitely many solutions

**Step 3: Geometric interpretation**
- Each solution corresponds to a point in the solution space
- The solution space is a hyperplane of dimension $m - n$
- Multiple points on this hyperplane satisfy the equation

**ML Applications:**
- **Underdetermined systems**: Common in high-dimensional ML
- **Regularization**: L1/L2 regularization to select unique solution
- **Feature selection**: Choose sparsest solution

**iv. [M] Pseudoinverse and solving singular systems**

**Problem**: When $A$ has no inverse, we need the pseudoinverse.

**Moore-Penrose Pseudoinverse:**
- **Definition**: $A^+$ is the pseudoinverse of $A$
- **Properties**: $AA^+A = A$, $A^+AA^+ = A^+$, $(AA^+)^T = AA^+$, $(A^+A)^T = A^+A$

**How to calculate pseudoinverse:**

**Method 1: SVD approach**
- $A = U\Sigma V^T$ (SVD decomposition)
- $A^+ = V\Sigma^+U^T$ where $\Sigma^+$ replaces non-zero singular values with their reciprocals

**Method 2: Normal equations**
- $A^+ = (A^TA)^+A^T$ (if $A^TA$ is invertible)
- $A^+ = A^T(AA^T)^+$ (if $AA^T$ is invertible)

**Step-by-Step Example:**

**Given**: $A = \begin{bmatrix} 1 & 2 \\ 1 & 2 \end{bmatrix}$ (singular matrix)

**Step 1: SVD decomposition**
- $A = U\Sigma V^T$ where $\Sigma = \begin{bmatrix} \sqrt{10} & 0 \\ 0 & 0 \end{bmatrix}$

**Step 2: Compute $\Sigma^+$**
- $\Sigma^+ = \begin{bmatrix} 1/\sqrt{10} & 0 \\ 0 & 0 \end{bmatrix}$

**Step 3: Compute pseudoinverse**
- $A^+ = V\Sigma^+U^T$

**ML Applications:**

**1. Linear Regression:**
- **Normal equations**: $\hat{\beta} = (X^TX)^+X^Ty$
- **Ridge regression**: $(X^TX + \lambda I)^{-1}X^Ty$
- **Handles multicollinearity**: Works even when $X^TX$ is singular

**2. Neural Networks:**
- **Weight initialization**: Avoid singular weight matrices
- **Gradient descent**: Use pseudoinverse for weight updates
- **Batch normalization**: Pseudoinverse for normalization parameters

**3. Dimensionality Reduction:**
- **PCA**: Pseudoinverse for projection matrices
- **LDA**: Pseudoinverse for discriminant analysis
- **Matrix factorization**: SVD-based approaches

**4. Optimization:**
- **Least squares**: $x = A^+b$ minimizes $||Ax - b||_2$
- **Regularization**: Add small values to diagonal
- **Numerical stability**: SVD is more stable than direct inversion

**Key Insight**: The pseudoinverse provides the "best" solution to linear systems even when the matrix is singular, making it essential for handling real-world ML problems with redundant or collinear features.

9. Derivatives are the backbone of gradient descent.

   i. [E] What does a derivative represent?

   ii. [M] What's the difference between a derivative, gradient, and Jacobian?

10. [H] Say we have the weights $w \in \mathbb{R}^{d \times m}$ and a mini-batch $x$ of $n$ elements, each element is of the shape $1 \times d$ so that $x \in \mathbb{R}^{n \times d}$. We have the output $y = f(x; w) = xw$. What is the dimension of the Jacobian $\frac{\partial y}{\partial x}$?

11. [H] Given a very large symmetric matrix $A$ that doesn't fit in memory, say $A \in \mathbb{R}^{1M \times 1M}$ and a function $f$ that can quickly compute $f(x) = Ax$ for $x \in \mathbb{R}^{1M}$. Find the unit vector $x$ such that $x^T Ax$ is minimal.
    
    **Hint:** Can you frame it as an optimization problem and use gradient descent to find an approximate solution?