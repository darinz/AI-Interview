# Dimensionality Reduction

Question 1. [E] Why do we need dimensionality reduction?

**Solution:**

**Intuitive Understanding:**
Think of dimensionality reduction like organizing a messy room. You have thousands of items (features), but you only need the most important ones to understand what's in the room. Dimensionality reduction helps you find and keep only the essential information.

**Mathematical Motivation:**
- **Curse of dimensionality**: As dimensions increase, data becomes sparse and distances become meaningless
- **Computational efficiency**: Fewer dimensions = faster algorithms
- **Visualization**: Humans can only visualize 2D/3D data
- **Overfitting prevention**: Fewer parameters reduce model complexity

**Step-by-Step Reasons:**

**1. Computational Efficiency:**
- **Problem**: High-dimensional data is expensive to process
- **Example**: Image with 1000×1000 pixels = 1 million features
- **Solution**: Reduce to 100 most important features
- **Benefit**: 10,000× faster computation

**2. Visualization:**
- **Problem**: Humans can only see 2D/3D plots
- **Example**: 50-dimensional dataset is impossible to visualize
- **Solution**: Project to 2D/3D while preserving structure
- **Benefit**: Understand data patterns visually

**3. Overfitting Prevention:**
- **Problem**: Too many features cause overfitting
- **Example**: 1000 features for 100 samples
- **Solution**: Reduce to 10-20 most relevant features
- **Benefit**: Better generalization to new data

**4. Noise Reduction:**
- **Problem**: High dimensions contain noise and irrelevant information
- **Example**: 1000 features, only 50 are meaningful
- **Solution**: Keep only the 50 meaningful ones
- **Benefit**: Cleaner, more robust models

**5. Storage and Memory:**
- **Problem**: Large datasets consume massive memory
- **Example**: 1TB dataset with 10,000 features
- **Solution**: Reduce to 100 features
- **Benefit**: 100× less storage needed

**ML Applications:**

**1. Principal Component Analysis (PCA):**
- **What**: Find directions of maximum variance
- **Why**: Remove redundant information
- **Example**: Face recognition - reduce 10,000 pixel features to 100 components

**2. Linear Discriminant Analysis (LDA):**
- **What**: Find directions that best separate classes
- **Why**: Improve classification performance
- **Example**: Email spam detection - reduce 50,000 word features to 20 discriminative features

**3. t-SNE:**
- **What**: Non-linear dimensionality reduction
- **Why**: Preserve local structure for visualization
- **Example**: Visualize high-dimensional embeddings in 2D

**4. Autoencoders:**
- **What**: Neural network that learns compressed representation
- **Why**: Learn non-linear relationships
- **Example**: Image compression and denoising

**Real-World Examples:**

**1. Image Processing:**
- **Raw data**: 1000×1000 pixel image = 1 million features
- **Reduced**: 100 principal components
- **Benefit**: 10,000× faster processing, same quality

**2. Text Analysis:**
- **Raw data**: 50,000 word vocabulary
- **Reduced**: 300-dimensional word embeddings
- **Benefit**: Capture semantic meaning, reduce noise

**3. Gene Expression:**
- **Raw data**: 20,000 genes per sample
- **Reduced**: 50 most important genes
- **Benefit**: Identify disease markers, reduce noise

**4. Recommendation Systems:**
- **Raw data**: User-item interaction matrix (millions × millions)
- **Reduced**: Low-rank approximation
- **Benefit**: Faster recommendations, better generalization

**Computational Considerations:**

**Memory Usage:**
- **Before**: $O(n \times d)$ where $n$ = samples, $d$ = features
- **After**: $O(n \times k)$ where $k \ll d$
- **Savings**: Often 100× to 1000× reduction

**Training Time:**
- **Before**: $O(d^2)$ or $O(d^3)$ algorithms
- **After**: $O(k^2)$ or $O(k^3)$ algorithms
- **Speedup**: Quadratic or cubic improvement

**Storage:**
- **Before**: Large feature matrices
- **After**: Compact representations
- **Benefit**: Easier to store and transfer

**Key Insight**: Dimensionality reduction is essential in modern ML because it addresses the fundamental problems of high-dimensional data: computational inefficiency, overfitting, noise, and the curse of dimensionality. It's not just about making algorithms faster—it's about making them work better by focusing on the most important information.

Question 2. [E] Eigendecomposition is a common factorization technique used for dimensionality reduction. Is the eigendecomposition of a matrix always unique?

**Solution:**

**Intuitive Understanding:**
Think of eigendecomposition like finding the "natural axes" of a matrix. Just like how a rectangle has two natural axes (length and width), a matrix has directions where it stretches or compresses vectors. The question is: are these directions always uniquely determined?

**Mathematical Definition:**
For a square matrix $A$, eigendecomposition is:
$$A = P \Lambda P^{-1}$$
where:
- $P$ = matrix of eigenvectors (columns)
- $\Lambda$ = diagonal matrix of eigenvalues
- $P^{-1}$ = inverse of eigenvector matrix

**Step-by-Step Analysis:**

**Step 1: When is eigendecomposition unique?**
- **Eigenvalues**: Always unique (up to ordering)
- **Eigenvectors**: NOT always unique

**Step 2: Cases where eigenvectors are unique:**
- **Distinct eigenvalues**: Each eigenvalue has exactly one eigenvector (up to scaling)
- **Example**: $`A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}`$
  - Eigenvalues: $\lambda_1 = 2, \lambda_2 = 3$
  - Eigenvectors: $`v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}`$
  - **Unique**: No other eigenvectors possible

**Step 3: Cases where eigenvectors are NOT unique:**
- **Repeated eigenvalues**: Multiple eigenvectors possible
- **Example**: $`A = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}`$ (identity matrix)
  - Eigenvalue: $\lambda = 2$ (repeated)
  - Eigenvectors: ANY non-zero vector is an eigenvector!
  - **Not unique**: Infinitely many choices

**Step 4: Geometric interpretation:**
- **Distinct eigenvalues**: Matrix stretches in different directions
- **Repeated eigenvalues**: Matrix stretches equally in multiple directions
- **Zero eigenvalue**: Matrix compresses some directions to zero

**Detailed Examples:**

**Example 1: Unique eigendecomposition**

$$A = \begin{bmatrix} 
3 & 1 \\ 
0 & 2 
\end{bmatrix}$$

**Step-by-step eigenvector calculation:**

**Step 1: Find eigenvalues**

Characteristic equation: $\det(A - \lambda I) = 0$

$$\det\begin{bmatrix} 
3-\lambda & 1 \\ 
0 & 2-\lambda 
\end{bmatrix} = (3-\lambda)(2-\lambda) = 0$$

**Eigenvalues**: $\lambda_1 = 3, \lambda_2 = 2$ (distinct)

**Step 2: Find eigenvector for $\lambda_1 = 3$**

Solve $(A - 3I)v_1 = 0$:

$$\begin{bmatrix} 
0 & 1 \\ 
0 & -1 \end{bmatrix}
\begin{bmatrix} 
x_1 \\ 
x_2 
\end{bmatrix} = 
\begin{bmatrix} 
0 \\ 
0 
\end{bmatrix}$$

From first row: $x_2 = 0$

From second row: $-x_2 = 0$ (redundant)

**Solution**: $`v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}`$ (any non-zero multiple)

**Step 3: Find eigenvector for $\lambda_2 = 2$**

Solve $(A - 2I)v_2 = 0$:

$$\begin{bmatrix} 
1 & 1 \\ 
0 & 0 
\end{bmatrix}
\begin{bmatrix} 
x_1 \\ 
x_2 
\end{bmatrix} = 
\begin{bmatrix} 
0 \\ 
0 
\end{bmatrix}$$

From first row: $x_1 + x_2 = 0$, so $x_1 = -x_2$

**Solution**: $`v_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}`$ (any non-zero multiple)

**Eigenvectors**: $`v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, v_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}`$

**Result**: Unique eigendecomposition

**Example 2: Non-unique eigendecomposition**

$$A = \begin{bmatrix} 
2 & 0 \\ 
0 & 2 
\end{bmatrix}$$

**Step-by-step eigenvector calculation:**

**Step 1: Find eigenvalues**

Characteristic equation: $\det(A - \lambda I) = 0$

$$\det\begin{bmatrix} 
2-\lambda & 0 \\ 
0 & 2-\lambda 
\end{bmatrix} = (2-\lambda)^2 = 0$$

**Eigenvalues**: $\lambda_1 = \lambda_2 = 2$ (repeated)

**Step 2: Find eigenvectors for $\lambda = 2$**

Solve $(A - 2I)v = 0$:

$$\begin{bmatrix} 
0 & 0 \\ 
0 & 0 
\end{bmatrix}
\begin{bmatrix} 
x_1 \\ 
x_2 
\end{bmatrix} = 
\begin{bmatrix} 
0 \\ 
0 
\end{bmatrix}$$

**Both equations**: $0 = 0$ (no constraints!)

**Solution**: ANY non-zero vector is an eigenvector!

**Examples of valid eigenvectors:**
- $`v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}`$
- $`v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}`$
- $`v_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}`$
- $`v_4 = \begin{bmatrix} 3 \\ -2 \end{bmatrix}`$
- **Any**: $`v = \begin{bmatrix} a \\ b \end{bmatrix}`$ where $`a^2 + b^2 \neq 0`$

**Eigenvectors**: ANY two linearly independent vectors!

**Result**: Infinitely many eigendecompositions

**Example 3: Partial uniqueness**

$$A = \begin{bmatrix} 
1 & 1 \\ 
0 & 1 
\end{bmatrix}$$

**Step-by-step eigenvector calculation:**

**Step 1: Find eigenvalues**

Characteristic equation: $\det(A - \lambda I) = 0$

$$\det\begin{bmatrix} 
1-\lambda & 1 \\ 
0 & 1-\lambda 
\end{bmatrix} = (1-\lambda)^2 = 0$$

**Eigenvalues**: $\lambda_1 = \lambda_2 = 1$ (repeated)

**Step 2: Find eigenvectors for $\lambda = 1$**

Solve $(A - I)v = 0$:

$$\begin{bmatrix} 
0 & 1 \\ 
0 & 0 
\end{bmatrix}
\begin{bmatrix} 
x_1 \\ 
x_2 
\end{bmatrix} = 
\begin{bmatrix} 
0 \\ 
0 
\end{bmatrix}$$
From first row: $x_2 = 0$
From second row: $0 = 0$ (redundant)
**Solution**: Only vectors of the form $v = \begin{bmatrix} a \\ 0 \end{bmatrix}$ where $a \neq 0$

**Step 3: Check if we have enough eigenvectors**
- **Algebraic multiplicity**: 2 (eigenvalue appears twice in characteristic polynomial)
- **Geometric multiplicity**: 1 (only one linearly independent eigenvector)
- **Problem**: We need 2 linearly independent eigenvectors for a 2×2 matrix

**Eigenvectors**: Only $`v = \begin{bmatrix} 1 \\ 0 \end{bmatrix}`$ (defective matrix)

**Result**: No complete eigendecomposition possible

**ML Applications and Implications:**

**1. Principal Component Analysis (PCA):**
- **Problem**: Non-unique eigenvectors for repeated eigenvalues
- **Solution**: Choose eigenvectors that maximize variance
- **Example**: If two eigenvalues are equal, choose orthogonal directions

**2. Linear Discriminant Analysis (LDA):**
- **Problem**: Non-unique discriminant directions
- **Solution**: Use regularization or choose specific directions
- **Example**: Add small values to diagonal for stability

**3. Spectral Clustering:**
- **Problem**: Non-unique eigenvectors affect clustering
- **Solution**: Use multiple eigenvectors or regularization
- **Example**: Combine several eigenvectors for robust clustering

**4. Matrix Factorization:**
- **Problem**: Non-unique factorizations
- **Solution**: Add constraints or use specific algorithms
- **Example**: Non-negative matrix factorization

**Computational Considerations:**

**Numerical Stability:**
- **Repeated eigenvalues**: Cause numerical instability
- **Solution**: Use SVD instead of eigendecomposition
- **Example**: SVD is more stable for near-repeated eigenvalues

**Algorithm Choice:**
- **Eigendecomposition**: When you need eigenvalues/eigenvectors
- **SVD**: When you need more stable decomposition
- **QR decomposition**: When you need orthogonal basis

**Regularization:**
- **Problem**: Singular or near-singular matrices
- **Solution**: Add $\lambda I$ to diagonal
- **Example**: $A + \lambda I$ where $\lambda$ is small

**Key Insight**: Eigendecomposition is NOT always unique, especially when eigenvalues are repeated. This is crucial in ML because it affects the stability and interpretability of algorithms like PCA and LDA. The solution is often to use regularization, choose specific eigenvectors, or use more stable alternatives like SVD.

Question 3. [M] Name some applications of eigenvalues and eigenvectors.

**Solution:**

**Intuitive Understanding:**
Eigenvalues and eigenvectors are like the "DNA" of a matrix - they tell us how the matrix transforms space. Just like how DNA determines how a cell behaves, eigenvalues and eigenvectors determine how a matrix behaves. They're everywhere in ML because they capture the fundamental structure of data and transformations.

**Mathematical Foundation:**
For a matrix $A$, if $Av = \lambda v$ where $v \neq 0$, then:
- $v$ is an eigenvector
- $\lambda$ is an eigenvalue
- This means $A$ only scales $v$ by factor $\lambda$

**Step-by-Step Applications:**

**1. Principal Component Analysis (PCA):**
- **What**: Find directions of maximum variance in data
- **How**: Eigenvectors of covariance matrix
- **Why**: Remove redundancy, reduce dimensions
- **Example**: Face recognition - reduce 10,000 pixel features to 100 components

**Mathematical Details:**
- **Covariance matrix**: $C = \frac{1}{n-1}X^T X$
- **Eigenvectors**: Directions of maximum variance
- **Eigenvalues**: Amount of variance in each direction
- **PCA transformation**: $Y = XW$ where $W$ contains top eigenvectors

**2. Linear Discriminant Analysis (LDA):**
- **What**: Find directions that best separate classes
- **How**: Eigenvectors of $S_w^{-1} S_b$ (within-class vs between-class scatter)
- **Why**: Improve classification performance
- **Example**: Email spam detection - find 20 best discriminative features

**Mathematical Details:**
- **Within-class scatter**: $S_w = \sum_{i=1}^c \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T$
- **Between-class scatter**: $S_b = \sum_{i=1}^c n_i (\mu_i - \mu)(\mu_i - \mu)^T$
- **Eigenvectors**: Directions that maximize class separation
- **Eigenvalues**: Measure of class separability

**3. Spectral Clustering:**
- **What**: Cluster data using graph Laplacian
- **How**: Eigenvectors of normalized Laplacian matrix
- **Why**: Handle non-convex clusters
- **Example**: Image segmentation - group similar pixels

**Mathematical Details:**
- **Graph Laplacian**: $L = D - W$ where $D$ is degree matrix, $W$ is adjacency matrix
- **Normalized Laplacian**: $L_{norm} = D^{-1/2} L D^{-1/2}$
- **Eigenvectors**: Embed data in low-dimensional space
- **Clustering**: Use k-means on embedded data

**4. PageRank Algorithm:**
- **What**: Rank web pages by importance
- **How**: Eigenvector of transition matrix
- **Why**: Find most influential pages
- **Example**: Google search ranking

**Mathematical Details:**
- **Transition matrix**: $P_{ij} = \frac{1}{\text{out-degree of page } j}$
- **PageRank vector**: $r = P^T r$ (eigenvector with eigenvalue 1)
- **Power iteration**: $r^{(k+1)} = P^T r^{(k)}$ until convergence

**5. Image Processing and Computer Vision:**
- **What**: Analyze image structure and features
- **How**: Eigenvectors of image covariance matrix
- **Why**: Extract important visual features
- **Example**: Face recognition, object detection

**Mathematical Details:**
- **Image matrix**: Each pixel is a feature
- **Covariance analysis**: Find common patterns
- **Eigenfaces**: Eigenvectors represent face components
- **Recognition**: Compare projections onto eigenfaces

**6. Recommendation Systems:**
- **What**: Find user-item relationships
- **How**: Eigenvectors of user-item interaction matrix
- **Why**: Discover latent factors
- **Example**: Netflix movie recommendations

**Mathematical Details:**
- **User-item matrix**: $R \in \mathbb{R}^{m \times n}$
- **Matrix factorization**: $R \approx U V^T$ where $U$ and $V$ are low-rank
- **Eigenvectors**: Represent latent user/item factors
- **Recommendation**: Predict ratings using factorized matrix

**7. Quantum Mechanics (ML Applications):**
- **What**: Model quantum systems
- **How**: Eigenvectors of Hamiltonian matrix
- **Why**: Find stable states and energy levels
- **Example**: Quantum machine learning, quantum optimization

**8. Network Analysis:**
- **What**: Analyze network structure
- **How**: Eigenvectors of adjacency matrix
- **Why**: Find influential nodes, communities
- **Example**: Social network analysis, protein interaction networks

**9. Signal Processing:**
- **What**: Analyze and filter signals
- **How**: Eigenvectors of signal covariance matrix
- **Why**: Remove noise, extract features
- **Example**: Speech recognition, EEG analysis

**10. Machine Learning Optimization:**
- **What**: Understand optimization landscapes
- **How**: Eigenvectors of Hessian matrix
- **Why**: Find optimal learning rates, avoid saddle points
- **Example**: Neural network training, hyperparameter tuning

**Real-World Examples:**

**1. Netflix Recommendation System:**
- **Problem**: Recommend movies to users
- **Solution**: Matrix factorization using eigenvectors
- **Result**: Personalized recommendations

**2. Google Search:**
- **Problem**: Rank web pages by relevance
- **Solution**: PageRank algorithm using eigenvectors
- **Result**: Most relevant pages appear first

**3. Face Recognition (Facebook):**
- **Problem**: Identify people in photos
- **Solution**: PCA using eigenfaces
- **Result**: Automatic face tagging

**4. Gene Expression Analysis:**
- **Problem**: Find genes related to diseases
- **Solution**: PCA to reduce 20,000 genes to 50 components
- **Result**: Identify disease markers

**5. Image Compression (JPEG):**
- **Problem**: Compress images efficiently
- **Solution**: DCT (Discrete Cosine Transform) using eigenvectors
- **Result**: 10:1 compression ratio

**Computational Considerations:**

**Eigenvalue Algorithms:**
- **Power iteration**: For largest eigenvalue
- **QR algorithm**: For all eigenvalues
- **Lanczos method**: For large sparse matrices
- **Randomized methods**: For approximate solutions

**Numerical Stability:**
- **Condition number**: $\frac{\lambda_{max}}{\lambda_{min}}$
- **Regularization**: Add $\lambda I$ for stability
- **SVD**: More stable than eigendecomposition

**Memory Usage:**
- **Dense matrices**: $O(n^2)$ storage
- **Sparse matrices**: $O(nnz)$ storage
- **Iterative methods**: $O(n)$ per iteration

**Key Insight**: Eigenvalues and eigenvectors are fundamental to understanding and manipulating data in ML. They capture the essential structure of matrices and transformations, making them indispensable for dimensionality reduction, clustering, ranking, and many other ML tasks. The key is choosing the right algorithm and handling numerical stability issues.

Question 4. [M] We want to do PCA on a dataset with multiple features in different ranges. For example, one feature is in the range 0-1 and another is in the range 10-1000. Will PCA work on this dataset?

**Solution:**

**Intuitive Understanding:**
Think of PCA like trying to find the "main direction" of a cloud of points. If one feature ranges from 0-1 and another from 10-1000, it's like trying to find the main direction of a cloud that's stretched 1000 times more in one direction than another. PCA will be dominated by the larger-scale feature and ignore the smaller one.

**The Problem:**
PCA is sensitive to the scale of features. Features with larger ranges will dominate the principal components, even if they're less important for the underlying pattern.

**Step-by-Step Analysis:**

**Step 1: What happens without scaling?**
- **Feature 1**: Range 0-1, variance ≈ 0.08
- **Feature 2**: Range 10-1000, variance ≈ 82,500
- **Result**: Feature 2 dominates PCA completely
- **Problem**: We lose information from Feature 1

**Step 2: Mathematical explanation:**
- **Covariance matrix**: $C = \frac{1}{n-1}X^T X$
- **Without scaling**: $C_{11} \ll C_{22}$ (Feature 1 variance << Feature 2 variance)
- **Eigenvalues**: $\lambda_1 \approx C_{22}$, $\lambda_2 \approx C_{11}$
- **First PC**: Almost entirely Feature 2 direction

**Step 3: Why this is bad:**
- **Information loss**: Feature 1 contributes almost nothing
- **Misleading results**: PCA suggests Feature 2 is most important
- **Poor visualization**: 2D plot shows only one dimension
- **Incorrect interpretation**: We miss the true data structure

**Detailed Example:**

**Dataset without scaling:**
```
Feature 1: [0.1, 0.2, 0.3, 0.4, 0.5]  (range: 0-1)
Feature 2: [100, 200, 300, 400, 500]  (range: 10-1000)
```

**Covariance matrix:**
$$C = \begin{bmatrix} 0.025 & 2.5 \\ 2.5 & 2500 \end{bmatrix}$$

**Eigenvalues**: $\lambda_1 = 2500.025$, $\lambda_2 = 0.025$
**First PC**: $[0.001, 0.999]$ (almost entirely Feature 2)

**Dataset with scaling:**
```
Feature 1: [0.1, 0.2, 0.3, 0.4, 0.5]  (standardized)
Feature 2: [100, 200, 300, 400, 500]  (standardized)
```

**Covariance matrix:**
$$C = \begin{bmatrix} 1.0 & 0.8 \\ 0.8 & 1.0 \end{bmatrix}$$

**Eigenvalues**: $\lambda_1 = 1.8$, $\lambda_2 = 0.2$
**First PC**: $[0.707, 0.707]$ (balanced contribution)

**Solutions:**

**1. Standardization (Z-score normalization):**
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Result**: All features have mean 0, variance 1
- **When to use**: When features have different scales but similar importance

**2. Min-Max scaling:**
- **Formula**: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
- **Result**: All features in range [0, 1]
- **When to use**: When you want to preserve the original distribution shape

**3. Robust scaling:**
- **Formula**: $x_{scaled} = \frac{x - \text{median}}{IQR}$
- **Result**: Less sensitive to outliers
- **When to use**: When data has outliers

**4. Unit vector scaling:**
- **Formula**: $x_{scaled} = \frac{x}{||x||_2}$
- **Result**: Each sample has unit length
- **When to use**: When you care about directions, not magnitudes

**ML Applications and Best Practices:**

**1. When to scale:**
- **Different units**: Height (cm) vs weight (kg)
- **Different ranges**: Age (0-100) vs income (0-1,000,000)
- **Different variances**: Some features much more variable
- **Before PCA**: Always scale before PCA

**2. When NOT to scale:**
- **Same units**: All features in same units and similar ranges
- **Interpretability**: When you want to preserve original scale
- **Tree-based models**: Random Forest, XGBoost are scale-invariant

**3. Real-world examples:**

**Example 1: Medical data**
- **Features**: Age (0-100), Blood pressure (80-200), Cholesterol (100-300)
- **Problem**: Different scales and units
- **Solution**: Standardize all features before PCA

**Example 2: Image data**
- **Features**: Pixel values (0-255)
- **Problem**: All same scale, but PCA still useful
- **Solution**: Can skip scaling, or normalize to [0,1]

**Example 3: Text data**
- **Features**: Word counts (0-1000), TF-IDF scores (0-1)
- **Problem**: Very different scales
- **Solution**: Standardize or use different scaling for each type

**Computational Considerations:**

**Memory usage:**
- **Before scaling**: Store original data
- **After scaling**: Store scaled data (same size)
- **PCA**: Compute covariance matrix of scaled data

**Numerical stability:**
- **Standardization**: More stable than min-max
- **Robust scaling**: Better with outliers
- **Check**: Ensure no division by zero

**Performance:**
- **Scaling time**: $O(n \times d)$ where $n$ = samples, $d$ = features
- **PCA time**: $O(d^3)$ for eigendecomposition
- **Total**: Scaling is usually fast compared to PCA

**Key Insight**: PCA will technically "work" on unscaled data, but it will give misleading results. The features with larger ranges will dominate the principal components, and you'll lose important information from smaller-scale features. Always scale your data before PCA to ensure all features contribute meaningfully to the analysis.

Question 5. [H] Under what conditions can one apply eigendecomposition? What about SVD?

   i. What is the relationship between SVD and eigendecomposition?

   ii. What's the relationship between PCA and SVD?

**Solution:**

**Intuitive Understanding:**
Think of eigendecomposition and SVD as two different ways to "break down" a matrix. Eigendecomposition is like finding the "natural axes" of a square matrix, while SVD is like finding the "natural axes" of any matrix (even rectangular ones). It's like the difference between analyzing a square room vs analyzing any shaped room.

**Mathematical Definitions:**

**Eigendecomposition:**
For a square matrix $A \in \mathbb{R}^{n \times n}$:
$$A = P \Lambda P^{-1}$$
where:
- $P$ = matrix of eigenvectors (columns)
- $\Lambda$ = diagonal matrix of eigenvalues
- $P^{-1}$ = inverse of eigenvector matrix

**SVD (Singular Value Decomposition):**
For any matrix $A \in \mathbb{R}^{m \times n}$:
$$A = U \Sigma V^T$$
where:
- $U \in \mathbb{R}^{m \times m}$ = left singular vectors (orthogonal)
- $\Sigma \in \mathbb{R}^{m \times n}$ = diagonal matrix of singular values
- $V \in \mathbb{R}^{n \times n}$ = right singular vectors (orthogonal)

**Step-by-Step Analysis:**

**Step 1: Conditions for Eigendecomposition**

**Requirements:**
- **Square matrix**: $A$ must be $n \times n$
- **Diagonalizable**: Must have $n$ linearly independent eigenvectors
- **Sufficient condition**: $n$ distinct eigenvalues
- **Necessary condition**: Geometric multiplicity = algebraic multiplicity for each eigenvalue

**When it works:**
- **Symmetric matrices**: Always diagonalizable
- **Normal matrices**: $AA^* = A^*A$ (e.g., Hermitian, unitary)
- **Matrices with distinct eigenvalues**: Always diagonalizable

**When it fails:**
- **Defective matrices**: Fewer than $n$ eigenvectors
- **Non-square matrices**: Cannot apply eigendecomposition
- **Example**: $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ (only one eigenvector)

**Step 2: Conditions for SVD**

**Requirements:**
- **Any matrix**: Works for $m \times n$ matrices
- **Always exists**: SVD always exists for any matrix
- **Numerically stable**: More stable than eigendecomposition

**When it works:**
- **Any matrix**: Real or complex, square or rectangular
- **Rank-deficient matrices**: Works even when matrix is singular
- **Ill-conditioned matrices**: More stable than eigendecomposition

**When it might be problematic:**
- **Very large matrices**: Computationally expensive
- **Sparse matrices**: May not preserve sparsity structure

**Step 3: Relationship between SVD and Eigendecomposition**

**For symmetric matrices:**
If $A$ is symmetric ($A = A^T$), then:
- **Eigendecomposition**: $A = P \Lambda P^T$ (orthogonal $P$)
- **SVD**: $A = U \Sigma V^T$ where $U = V = P$ and $\Sigma = |\Lambda|$
- **Key insight**: For symmetric matrices, SVD and eigendecomposition are essentially the same

**For general matrices:**
- **Eigendecomposition**: $A = P \Lambda P^{-1}$ (if it exists)
- **SVD**: $A = U \Sigma V^T$ (always exists)
- **Relationship**: $U$ and $V$ are orthogonal, $P$ is not necessarily orthogonal

**Mathematical connection:**
- **Eigenvalues of $A^T A$**: Equal to squares of singular values of $A$
- **Eigenvectors of $A^T A$**: Equal to right singular vectors of $A$
- **Eigenvectors of $A A^T$**: Equal to left singular vectors of $A$

**Step 4: Relationship between PCA and SVD**

**PCA using Eigendecomposition:**
- **Covariance matrix**: $C = \frac{1}{n-1}X^T X$
- **Eigendecomposition**: $C = P \Lambda P^T$
- **Principal components**: Columns of $P$
- **Explained variance**: Diagonal elements of $\Lambda$

**PCA using SVD:**
- **Centered data**: $X_{centered} = X - \mu$
- **SVD**: $X_{centered} = U \Sigma V^T$
- **Principal components**: Columns of $V$
- **Explained variance**: $\frac{\sigma_i^2}{n-1}$ where $\sigma_i$ are singular values

**Key insight**: PCA can be computed using either eigendecomposition of the covariance matrix or SVD of the centered data matrix. SVD is often preferred because it's more numerically stable.

**Detailed Examples:**

**Example 1: Symmetric matrix**
$$A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$$

**Eigendecomposition:**
- **Eigenvalues**: $\lambda_1 = 4, \lambda_2 = 2$
- **Eigenvectors**: $v_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}, v_2 = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix}$
- **Result**: $A = P \Lambda P^T$ where $P$ is orthogonal

**SVD:**
- **Singular values**: $\sigma_1 = 4, \sigma_2 = 2$
- **Left singular vectors**: Same as eigenvectors
- **Right singular vectors**: Same as eigenvectors
- **Result**: $A = U \Sigma V^T$ where $U = V = P$

**Example 2: Non-square matrix**
$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$$

**Eigendecomposition**: Not possible (not square)

**SVD:**
- **Singular values**: $\sigma_1 = 9.525, \sigma_2 = 0.514$
- **Left singular vectors**: $U \in \mathbb{R}^{3 \times 3}$
- **Right singular vectors**: $V \in \mathbb{R}^{2 \times 2}$
- **Result**: $A = U \Sigma V^T$ (always possible)

**ML Applications and Practical Considerations:**

**1. When to use Eigendecomposition:**
- **Symmetric matrices**: Covariance matrices, correlation matrices
- **Square matrices**: When you need eigenvalues and eigenvectors
- **Interpretability**: When you want to understand the "natural directions"

**2. When to use SVD:**
- **Any matrix**: Rectangular matrices, rank-deficient matrices
- **Numerical stability**: When eigendecomposition might be unstable
- **PCA**: Often preferred for PCA computation
- **Matrix approximation**: Low-rank approximations

**3. Computational considerations:**

**Eigendecomposition:**
- **Complexity**: $O(n^3)$ for $n \times n$ matrix
- **Memory**: $O(n^2)$ storage
- **Stability**: Can be unstable for repeated eigenvalues

**SVD:**
- **Complexity**: $O(mn^2)$ for $m \times n$ matrix
- **Memory**: $O(mn)$ storage
- **Stability**: More numerically stable

**4. Real-world examples:**

**Example 1: PCA for face recognition**
- **Data**: 1000×1000 pixel images
- **Matrix**: 1000×1000 covariance matrix
- **Method**: Eigendecomposition (symmetric matrix)
- **Result**: Eigenfaces for recognition

**Example 2: Recommendation systems**
- **Data**: User-item rating matrix (1000×10000)
- **Matrix**: Rectangular, sparse
- **Method**: SVD (non-square matrix)
- **Result**: Low-rank approximation for recommendations

**Example 3: Text analysis**
- **Data**: Document-term matrix (1000×50000)
- **Matrix**: Rectangular, sparse
- **Method**: SVD (non-square matrix)
- **Result**: Latent semantic analysis

**Key Insight**: Eigendecomposition and SVD are both powerful matrix factorization techniques, but they have different strengths. Eigendecomposition is great for symmetric matrices and when you need eigenvalues/eigenvectors, while SVD is more general and numerically stable. For PCA, SVD is often preferred because it's more stable and works directly with the data matrix rather than the covariance matrix.

Question 6. [H] How does t-SNE (T-distributed Stochastic Neighbor Embedding) work? Why do we need it?

**Solution:**

**Intuitive Understanding:**
Think of t-SNE like creating a "map" of your high-dimensional data. Just like how a world map preserves the relative distances between cities, t-SNE creates a 2D or 3D map that preserves the local neighborhood structure of your data. It's like taking a complex, multi-dimensional dataset and creating a simple map that you can actually look at and understand.

**Why do we need t-SNE?**

**1. Visualization of High-Dimensional Data:**
- **Problem**: Humans can only visualize 2D/3D data
- **Solution**: t-SNE projects high-dimensional data to 2D/3D
- **Example**: Visualize 1000-dimensional gene expression data in 2D

**2. Understanding Data Structure:**
- **Problem**: High-dimensional data has hidden structure
- **Solution**: t-SNE reveals clusters, patterns, and relationships
- **Example**: Discover patient subgroups in medical data

**3. Exploratory Data Analysis:**
- **Problem**: Need to understand data before modeling
- **Solution**: t-SNE provides intuitive data overview
- **Example**: Explore customer segments before building recommendation system

**4. Model Validation:**
- **Problem**: Need to verify that models capture data structure
- **Solution**: Visualize model outputs with t-SNE
- **Example**: Check if neural network learned meaningful representations

**Mathematical Foundation:**

**Step 1: Convert distances to probabilities**
For each pair of points $(i, j)$ in high-dimensional space:
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2/2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2/2\sigma_i^2)}$$

Where:
- $x_i, x_j$ are high-dimensional points
- $\sigma_i$ is the bandwidth parameter for point $i$
- $p_{j|i}$ is the probability that $j$ is a neighbor of $i$

**Step 2: Symmetrize the probabilities**
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

Where:
- $n$ is the number of data points
- $p_{ij}$ is the symmetric probability

**Step 3: Define probabilities in low-dimensional space**
For each pair of points $(i, j)$ in low-dimensional space:
$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

Where:
- $y_i, y_j$ are low-dimensional points
- $q_{ij}$ is the probability that $j$ is a neighbor of $i$ in low-dimensional space

**Step 4: Minimize the Kullback-Leibler divergence**
$$KL(P||Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

Where:
- $P$ is the probability distribution in high-dimensional space
- $Q$ is the probability distribution in low-dimensional space
- $KL(P||Q)$ measures how different the two distributions are

**Step-by-Step Algorithm:**

**Step 1: Initialize low-dimensional points**
- **Random initialization**: Start with random 2D/3D points
- **PCA initialization**: Use PCA to get better starting point
- **Example**: 1000 points in 2D space

**Step 2: Compute high-dimensional probabilities**
- **For each point**: Find its neighbors in high-dimensional space
- **Gaussian kernel**: Use Gaussian distribution to compute probabilities
- **Adaptive bandwidth**: Adjust $\sigma_i$ for each point based on local density

**Step 3: Compute low-dimensional probabilities**
- **t-distribution**: Use t-distribution (heavier tails) for low-dimensional space
- **Why t-distribution**: Prevents crowding problem, allows better separation

**Step 4: Gradient descent optimization**
- **Gradient**: Compute gradient of KL divergence
- **Update rule**: $y_i^{(t+1)} = y_i^{(t)} + \eta \frac{\partial KL}{\partial y_i} + \alpha(t)(y_i^{(t)} - y_i^{(t-1)})$
- **Momentum**: Use momentum to avoid local minima

**Step 5: Repeat until convergence**
- **Iterations**: Typically 1000-5000 iterations
- **Convergence**: When KL divergence stops decreasing significantly

**Key Parameters:**

**1. Perplexity:**
- **What**: Controls the number of neighbors for each point
- **Range**: Typically 5-50
- **Effect**: Higher perplexity = more global structure, lower perplexity = more local structure
- **Example**: Perplexity 30 means each point considers ~30 neighbors

**2. Learning rate:**
- **What**: Controls step size in gradient descent
- **Range**: Typically 10-1000
- **Effect**: Higher learning rate = faster convergence but may overshoot
- **Example**: Learning rate 200 is common starting point

**3. Number of iterations:**
- **What**: How many optimization steps to run
- **Range**: Typically 1000-5000
- **Effect**: More iterations = better convergence but slower computation
- **Example**: 1000 iterations for quick exploration, 5000 for final results

**4. Early exaggeration:**
- **What**: Multiplies $p_{ij}$ by a factor in early iterations
- **Range**: Typically 4-12
- **Effect**: Helps separate clusters in early stages
- **Example**: Early exaggeration 4 for first 100 iterations

**ML Applications:**

**1. Data Visualization:**
- **Gene expression**: Visualize 20,000 genes in 2D
- **Image features**: Visualize CNN features
- **Text embeddings**: Visualize word/document embeddings
- **Customer data**: Visualize customer segments

**2. Exploratory Data Analysis:**
- **Clustering validation**: Check if clusters make sense
- **Outlier detection**: Identify unusual data points
- **Data quality**: Understand data distribution
- **Feature engineering**: Guide feature selection

**3. Model Interpretation:**
- **Neural networks**: Visualize learned representations
- **Clustering**: Understand cluster structure
- **Classification**: Visualize decision boundaries
- **Anomaly detection**: Visualize normal vs anomalous data

**4. Research and Development:**
- **Algorithm comparison**: Compare different algorithms visually
- **Hyperparameter tuning**: Understand parameter effects
- **Data preprocessing**: Guide preprocessing decisions
- **Model selection**: Choose between different models

**Real-World Examples:**

**1. Single-Cell RNA Sequencing:**
- **Problem**: Analyze 20,000 genes across 10,000 cells
- **Solution**: t-SNE to visualize cell types in 2D
- **Result**: Discover new cell types and developmental trajectories

**2. Image Classification:**
- **Problem**: Understand what CNN learns
- **Solution**: t-SNE on CNN features
- **Result**: Visualize how CNN groups similar images

**3. Customer Segmentation:**
- **Problem**: Understand customer behavior patterns
- **Solution**: t-SNE on customer features
- **Result**: Identify distinct customer segments

**4. Drug Discovery:**
- **Problem**: Find similar chemical compounds
- **Solution**: t-SNE on molecular fingerprints
- **Result**: Group compounds by similarity

**Advantages and Limitations:**

**Advantages:**
- **Non-linear**: Captures complex, non-linear relationships
- **Local structure**: Preserves local neighborhoods well
- **Visualization**: Creates beautiful, interpretable plots
- **Flexibility**: Works with any distance metric

**Limitations:**
- **Computational cost**: Expensive for large datasets (>10,000 points)
- **Non-deterministic**: Results vary between runs
- **Global structure**: May not preserve global distances
- **Parameters**: Sensitive to parameter choices

**Computational Considerations:**

**Memory usage:**
- **Distance matrix**: $O(n^2)$ for $n$ points
- **Probability matrix**: $O(n^2)$ storage
- **Total**: $O(n^2)$ memory complexity

**Time complexity:**
- **Distance computation**: $O(n^2 \times d)$ where $d$ is dimension
- **Probability computation**: $O(n^2)$
- **Gradient descent**: $O(n^2)$ per iteration
- **Total**: $O(n^2 \times \text{iterations})$

**Optimization techniques:**
- **Barnes-Hut approximation**: Reduce complexity to $O(n \log n)$
- **FIt-SNE**: Faster implementation using FFT
- **GPU acceleration**: Use GPUs for faster computation

**Key Insight**: t-SNE is a powerful tool for visualizing and understanding high-dimensional data. It's particularly useful for exploratory data analysis, model interpretation, and discovering hidden patterns in data. However, it's computationally expensive and results can vary between runs, so it's best used as a complementary tool alongside other analysis methods.