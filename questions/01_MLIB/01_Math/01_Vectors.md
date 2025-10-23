
## 1. Dot product

    i. [E] What's the geometric interpretation of the dot product of two vectors?
    
    **Solution:** The dot product is fundamental in ML and has several key interpretations:
    - **Similarity measure**: $a \cdot b = |a| \cdot |b| \cos(\theta)$ measures how "aligned" two vectors are
    - **Attention mechanisms**: In transformers, attention scores are computed as dot products between query and key vectors
    - **Feature similarity**: Used in recommendation systems to find similar users/items based on feature vectors
    - **Neural network computations**: Linear layers compute $Wx + b$ where $Wx$ involves dot products
    - **Cosine similarity**: $\cos(\theta) = \frac{a \cdot b}{|a| \cdot |b|}$ is crucial for text similarity, collaborative filtering
    - **Orthogonality test**: Zero dot product indicates uncorrelated features (important for feature selection)
    
    ii. [E] Given a vector $u$, find vector $v$ of unit length such that the dot product of $u$ and $v$ is maximum.
    
    **Solution:** This optimization problem is common in ML, particularly in:
    - **Gradient descent**: Finding the direction of steepest ascent
    - **Principal Component Analysis (PCA)**: Finding the direction of maximum variance
    - **Neural network optimization**: Direction of maximum change in loss function
    
    **Mathematical solution:**
    - The dot product $u \cdot v = |u| \cdot |v| \cos(\theta) = |u| \cos(\theta)$ (since $|v| = 1$)
    - This is maximized when $\cos(\theta) = 1$, which occurs when $\theta = 0$ (vectors are parallel)
    - Therefore, $v$ should be in the same direction as $u$
    - The unit vector in the direction of $u$ is: $v = \frac{u}{|u|}$
    - Maximum dot product value: $u \cdot v = |u|$
    
    **ML Applications:**
    - **Gradient normalization**: In training neural networks, gradients are often normalized to unit length
    - **Feature alignment**: In contrastive learning, we align positive pairs by maximizing their dot product
    - **Attention mechanisms**: The optimal attention direction aligns with the query vector

## 2. Outer product
   
   i. [E] Given two vectors $a = [3, 2, 1]$ and $b = [-1, 0, 1]$. Calculate the outer product $a^T b$?
   
   **Solution:** The outer product $a \otimes b$ (or $ab^T$) is calculated as:
   $$a \otimes b = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 3(-1) & 3(0) & 3(1) \\ 2(-1) & 2(0) & 2(1) \\ 1(-1) & 1(0) & 1(1) \end{bmatrix} = \begin{bmatrix} -3 & 0 & 3 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$
   
   ii. [M] Give an example of how the outer product can be useful in ML.
   
   **Solution:** The outer product is crucial in ML for modeling interactions:
   - **Factorization Machines**: Use outer products to capture feature interactions in recommendation systems
   - **Neural Tensor Networks**: Model relationships between entities using outer products
   - **Bilinear pooling**: In computer vision, outer products capture spatial feature interactions
   - **Kernel methods**: Polynomial kernels $K(x,y) = (x^T y)^d$ involve outer products for feature interactions
   - **Attention mechanisms**: Self-attention computes $QK^T$ where each element is an outer product
   - **Covariance estimation**: For PCA, we compute $\frac{1}{n}XX^T$ (outer product of data matrix with itself)
   - **Word embeddings**: In NLP, outer products model word co-occurrence patterns

## 3. [E] What does it mean for two vectors to be linearly independent?

**Solution:** Linear independence is crucial in ML for understanding feature relationships and model capacity:

**Definition**: Two vectors $\vec{v_1}$ and $\vec{v_2}$ are linearly independent if no non-trivial linear combination equals zero:
- **Mathematical condition**: The only solution to $c_1\vec{v_1} + c_2\vec{v_2} = \vec{0}$ is $c_1 = c_2 = 0$
- **Practical test**: One vector is not a scalar multiple of the other

**ML Applications:**
- **Feature selection**: Linearly dependent features provide redundant information and can be removed
- **Dimensionality reduction**: PCA identifies linearly independent directions of maximum variance
- **Overfitting prevention**: Too many linearly dependent features can cause overfitting
- **Model interpretability**: Independent features have clearer individual contributions
- **Regularization**: L1 regularization encourages linear independence by driving some coefficients to zero

**Computational considerations:**
- **Rank of feature matrix**: Number of linearly independent features = rank of feature matrix
- **Condition number**: Near-linear dependence causes numerical instability in optimization
- **Feature engineering**: Create new features that are linearly independent of existing ones

**Examples:**
- **Independent**: Age and income (generally uncorrelated)
- **Dependent**: Height in cm and height in inches (perfectly correlated)

4. [M] Given two sets of vectors $A = a_1, a_2, a_3, \cdots , a_n$ and $B = b_1, b_2, b_3, \cdots , b_m$. How do you check that they share the same basis?

**Solution:** This problem is fundamental in ML for comparing feature spaces and understanding model representations:

**ML Context:**
- **Feature space comparison**: Do two different feature engineering approaches capture the same information?
- **Model interpretability**: Do different neural network layers learn equivalent representations?
- **Transfer learning**: Are pre-trained features sufficient for a new task?
- **Dimensionality reduction**: Does PCA preserve the essential information in the original features?

**Practical Methods:**

**Method 1: Rank comparison (most efficient)**
- Check if rank([A, B]) = rank(A) = rank(B)
- If true, then span(A) = span(B)
- **ML interpretation**: No new information is gained by combining the feature sets

**Method 2: Projection test**
- Project each vector in A onto span(B) and vice versa
- If all projections equal the original vectors, they span the same space
- **ML application**: Test if one feature set can reconstruct another

**Method 3: SVD-based approach**
- Compute SVD of [A | B] and check if the singular values indicate the same effective dimensionality
- **ML benefit**: Handles numerical precision issues common in real datasets

**Computational considerations for ML:**
- **Numerical stability**: Use SVD instead of Gaussian elimination for large, ill-conditioned matrices
- **Efficiency**: For high-dimensional data, use randomized SVD or iterative methods
- **Regularization**: Add small regularization terms to handle near-singular matrices

**Real-world example:**
- Compare word embeddings from Word2Vec vs. GloVe
- Check if different CNN layers capture equivalent visual features

5. [M] Given $n$ vectors, each of $d$ dimensions. What is the dimension of their span?

**Solution:** Span dimension is crucial in ML for understanding model capacity and feature space complexity:

**ML Context:**
- **Model capacity**: Higher span dimension = more expressive model
- **Feature engineering**: How many independent features do we actually have?
- **Representation learning**: What's the intrinsic dimensionality of learned representations?
- **Overfitting**: Too many features relative to span dimension can cause overfitting

**Key principles:**
- **Maximum possible dimension**: $\min(n, d)$
  - Cannot exceed the number of vectors ($n$)
  - Cannot exceed the dimension of the ambient space ($d$)

**ML Applications:**

**Feature space analysis:**
- **Effective dimensionality**: Real datasets often have much lower intrinsic dimension than ambient dimension
- **PCA components**: Number of significant principal components = span dimension of centered data
- **Feature selection**: Remove redundant features to reduce span dimension

**Neural network analysis:**
- **Layer capacity**: Each layer's span dimension determines its representational power
- **Bottleneck layers**: Span dimension < input dimension creates compression
- **Representation learning**: Autoencoders learn low-dimensional representations

**Computational methods:**
1. **SVD**: $\text{rank}(X) = \text{number of non-zero singular values}$
2. **QR decomposition**: Number of non-zero diagonal elements in R
3. **Eigenvalue analysis**: For covariance matrices, count non-zero eigenvalues

**Real-world examples:**
- **Image data**: 1000×1000 pixel images might have span dimension ~100-1000
- **Text embeddings**: 300D word vectors might span only 50-200 dimensions
- **Recommendation systems**: User-item matrices are typically low-rank

**Practical computation:**
$$\text{dim}(\text{span}\{v_1, v_2, \ldots, v_n\}) = \text{rank}([v_1, v_2, \ldots, v_n])$$

6. Norms and metrics
   
   i. [E] What's a norm? What are $L_0, L_1, L_2, L_{\infty}$ norms?
   
   **Solution:**

   **What is a norm?**
   A norm measures vector "size" and is fundamental in ML for regularization, optimization, and distance calculations:
   1. **Positive definiteness**: $||v|| \geq 0$ and $||v|| = 0$ iff $v = 0$
   2. **Homogeneity**: $||\alpha v|| = |\alpha| \cdot ||v||$ for scalar $\alpha$
   3. **Triangle inequality**: $||u + v|| \leq ||u|| + ||v||$

   **ML Applications of Common Norms:**

   **$L_0$ norm (Sparsity)**: $||x||_0 = \text{number of non-zero elements}$
   - **Feature selection**: Minimize $L_0$ to select most important features
   - **Sparse coding**: Find sparse representations of data
   - **Not differentiable**: Use $L_1$ as convex relaxation

   **$L_1$ norm (Lasso regularization)**: $||x||_1 = \sum_{i=1}^n |x_i|$
   - **L1 regularization**: $\min \text{loss} + \lambda ||w||_1$ encourages sparsity
   - **Robust to outliers**: Less sensitive than $L_2$ to extreme values
   - **Feature selection**: Automatically drives some weights to zero

   **$L_2$ norm (Ridge regularization)**: $||x||_2 = \sqrt{\sum_{i=1}^n x_i^2}$
   - **L2 regularization**: $\min \text{loss} + \lambda ||w||_2^2$ prevents overfitting
   - **Gradient descent**: $L_2$ gradients are well-behaved for optimization
   - **Distance metrics**: Euclidean distance for clustering, similarity

   **$L_{\infty}$ norm (Max pooling)**: $||x||_{\infty} = \max_{i} |x_i|$
   - **Max pooling**: In CNNs, captures strongest activation
   - **Robust optimization**: Minimize worst-case loss
   - **Normalization**: Used in batch normalization for stability
   
   ii. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

   **Solution:**

   **ML Context - Norms vs Metrics:**
   - **Norms**: Measure vector "size" for regularization, optimization, feature scaling
   - **Metrics**: Measure "distance" for clustering, similarity, loss functions
   - **Key difference**: Norms need vector space structure, metrics work on any data

   **From norm to metric (always possible in ML):**
   - **Distance functions**: $d(x,y) = ||x - y||$ creates distance from any norm
   - **ML applications**:
     - **$L_2$ norm → Euclidean distance**: Used in k-NN, clustering, similarity
     - **$L_1$ norm → Manhattan distance**: Robust to outliers, used in sparse data
     - **$L_{\infty}$ norm → Chebyshev distance**: Used in max pooling, robust optimization

   **From metric to norm (requires vector space structure):**
   - **Not always possible**: Only if metric has vector space properties
   - **Required conditions**:
     1. **Translation invariance**: $d(x + z, y + z) = d(x, y)$ (shift doesn't change distance)
     2. **Homogeneity**: $d(\alpha x, \alpha y) = |\alpha| d(x, y)$ (scaling preserves relative distances)
     3. **Origin condition**: $d(0, x) = ||x||$ defines the norm

   **ML Examples:**
   - **Euclidean metric → $L_2$ norm**: $d(x,y) = \sqrt{\sum (x_i - y_i)^2}$ gives $||x||_2$
   - **Manhattan metric → $L_1$ norm**: $d(x,y) = \sum |x_i - y_i|$ gives $||x||_1$
   - **Cosine distance**: Cannot define a norm (not translation invariant)
   - **Jaccard distance**: Cannot define a norm (not defined on vector space)

   **Practical ML considerations:**
   - **Feature scaling**: Use norms to normalize features before applying distance-based algorithms
   - **Regularization**: Use norms in loss functions to control model complexity
   - **Optimization**: Different norms have different optimization landscapes
