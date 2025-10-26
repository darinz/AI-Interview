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
- **Example**: $A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$
  - Eigenvalues: $\lambda_1 = 2, \lambda_2 = 3$
  - Eigenvectors: $v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
  - **Unique**: No other eigenvectors possible

**Step 3: Cases where eigenvectors are NOT unique:**
- **Repeated eigenvalues**: Multiple eigenvectors possible
- **Example**: $A = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$ (identity matrix)
  - Eigenvalue: $\lambda = 2$ (repeated)
  - Eigenvectors: ANY non-zero vector is an eigenvector!
  - **Not unique**: Infinitely many choices

**Step 4: Geometric interpretation:**
- **Distinct eigenvalues**: Matrix stretches in different directions
- **Repeated eigenvalues**: Matrix stretches equally in multiple directions
- **Zero eigenvalue**: Matrix compresses some directions to zero

**Detailed Examples:**

**Example 1: Unique eigendecomposition**
$$A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$$

**Eigenvalues**: $\lambda_1 = 3, \lambda_2 = 2$ (distinct)
**Eigenvectors**: $v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, v_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$
**Result**: Unique eigendecomposition

**Example 2: Non-unique eigendecomposition**
$$A = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

**Eigenvalues**: $\lambda_1 = \lambda_2 = 2$ (repeated)
**Eigenvectors**: ANY two linearly independent vectors!
**Result**: Infinitely many eigendecompositions

**Example 3: Partial uniqueness**
$$A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

**Eigenvalues**: $\lambda_1 = \lambda_2 = 1$ (repeated)
**Eigenvectors**: Only $v = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (defective matrix)
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

Question 5. [H] Under what conditions can one apply eigendecomposition? What about SVD?

   i. What is the relationship between SVD and eigendecomposition?

   ii. What's the relationship between PCA and SVD?

Question 6. [H] How does t-SNE (T-distributed Stochastic Neighbor Embedding) work? Why do we need it?