# Vectors

## 1. Dot product

### Question i. [E] What's the geometric interpretation of the dot product of two vectors?

**Solution:** The dot product is fundamental in ML and has several key interpretations:

**Intuitive Understanding:**
Think of the dot product as measuring how much two vectors "point in the same direction." If two vectors are perfectly aligned, the dot product is large and positive. If they point in opposite directions, it's large and negative. If they're perpendicular, it's zero.

**Mathematical Foundation:**
The dot product formula $a \cdot b = |a| \cdot |b| \cos(\theta)$ tells us:
- **$|a| \cdot |b|$**: Maximum possible value (when vectors are parallel)
- **$\cos(\theta)$**: How much of this maximum we actually get
- **$\theta$**: Angle between vectors (0° = parallel, 90° = perpendicular, 180° = opposite)

**ML Applications Explained:**

**1. Similarity Measure**: 
- **Intuition**: Like measuring how similar two people's preferences are
- **Example**: User vectors [1,0,1,0] and [1,0,0,1] have dot product 1 (somewhat similar)
- **ML Use**: Recommendation systems use this to find users with similar tastes

**2. Attention Mechanisms**:
- **Intuition**: "How much should I pay attention to this word/feature?"
- **Process**: Query vector asks "what am I looking for?" Key vector says "this is what I contain"
- **Result**: High dot product = high attention = "this is relevant!"

**3. Neural Network Computations**:
- **Intuition**: Each neuron computes a weighted sum of inputs
- **Formula**: $Wx + b$ where $Wx$ is a dot product between weight vector and input vector
- **Example**: If weights are [0.5, -0.3, 0.8] and inputs are [1, 0, 1], result is 0.5×1 + (-0.3)×0 + 0.8×1 = 1.3

**4. Cosine Similarity**:
- **Why useful**: Measures direction similarity regardless of magnitude
- **Formula**: $\cos(\theta) = \frac{a \cdot b}{|a| \cdot |b|}$
- **Example**: Text similarity - "cat" and "kitten" have high cosine similarity even if one appears more frequently

**5. Orthogonality Test**:
- **Intuition**: Perpendicular vectors are "completely different"
- **ML Use**: Feature selection - remove features that are too similar (low dot product with target)

### Question ii. [E] Given a vector $u$, find vector $v$ of unit length such that the dot product of $u$ and $v$ is maximum.

**Solution:** This optimization problem is common in ML, particularly in:

**Intuitive Understanding:**
Imagine you have a flashlight (vector $u$) and want to point it in the direction that gives maximum "light" (dot product). The answer is obvious: point it in the same direction as $u$! This is like asking "which way should I look to see the most of this object?"

**Step-by-Step Mathematical Solution:**

**Step 1: Set up the problem**
- We want to maximize $u \cdot v$ where $|v| = 1$ (unit length constraint)
- The dot product formula: $u \cdot v = |u| \cdot |v| \cos(\theta) = |u| \cos(\theta)$ (since $|v| = 1$)

**Step 2: Find the maximum**
- Since $|u|$ is fixed, we need to maximize $\cos(\theta)$
- $\cos(\theta)$ is maximized when $\theta = 0$ (cosine of 0° = 1)
- This means the vectors are parallel (pointing in the same direction)

**Step 3: Find the solution**
- $v$ should point in the same direction as $u$
- To make it unit length: $v = \frac{u}{|u|}$ (divide by its length)
- Maximum dot product value: $u \cdot v = |u| \cdot 1 = |u|$

**Concrete Example:**
- If $u = [3, 4]$, then $|u| = 5$
- The unit vector in same direction: $v = \frac{[3,4]}{5} = [0.6, 0.8]$
- Maximum dot product: $[3,4] \cdot [0.6,0.8] = 3(0.6) + 4(0.8) = 1.8 + 3.2 = 5 = |u|$

**ML Applications Explained:**

**1. Gradient Descent:**
- **Problem**: Which direction should we move to increase the loss function fastest?
- **Solution**: Move in the direction of the gradient (steepest ascent)
- **Why**: The gradient points in the direction of maximum increase

**2. PCA (Principal Component Analysis):**
- **Problem**: Which direction captures the most variance in the data?
- **Solution**: The first principal component is the direction of maximum variance
- **Why**: We want to project data onto the direction that preserves the most information

**3. Neural Network Training:**
- **Problem**: How should we update weights to minimize loss?
- **Solution**: Move in the direction opposite to the gradient (steepest descent)
- **Why**: The negative gradient points toward the minimum

**4. Attention Mechanisms:**
- **Problem**: Which parts of the input should get the most attention?
- **Solution**: Query vectors that align best with key vectors get high attention
- **Why**: High dot product means high relevance/similarity

## 2. Outer product
   
Question i. [E] Given two vectors $a = [3, 2, 1]$ and $b = [-1, 0, 1]$. Calculate the outer product $a^T b$?

**Solution:** Let's calculate the outer product step by step:

**Understanding the Outer Product:**
The outer product creates a matrix by multiplying every element of the first vector with every element of the second vector. Think of it as "spreading out" one vector against another.

**Step-by-Step Calculation:**

**Given vectors:**
- $a = [3, 2, 1]$ (column vector)
- $b = [-1, 0, 1]$ (row vector)

**Step 1: Set up the multiplication**
$$a \otimes b = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 0 & 1 \end{bmatrix}$$

**Step 2: Multiply each element of a with each element of b**
- Row 1: $3 \times [-1, 0, 1] = [3(-1), 3(0), 3(1)] = [-3, 0, 3]$
- Row 2: $2 \times [-1, 0, 1] = [2(-1), 2(0), 2(1)] = [-2, 0, 2]$
- Row 3: $1 \times [-1, 0, 1] = [1(-1), 1(0), 1(1)] = [-1, 0, 1]$

**Step 3: Combine into matrix**
$$a \otimes b = \begin{bmatrix} -3 & 0 & 3 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

**Intuitive Understanding:**
- The outer product creates a "multiplication table" between two vectors
- Each element $(i,j)$ of the result is $a_i \times b_j$
- Notice the pattern: each row is a scaled version of vector $b$
- The scaling factor is the corresponding element of vector $a$

Question ii. [M] Give an example of how the outer product can be useful in ML.

**Solution:** The outer product is crucial in ML for modeling interactions:

**Intuitive Understanding:**
The outer product captures "what happens when every element of one thing interacts with every element of another thing." It's like creating a comprehensive interaction table.

**Detailed ML Applications:**

**1. Factorization Machines (Recommendation Systems):**
- **Problem**: How do user preferences interact with item features?
- **Solution**: Use outer product to model user-item interactions
- **Example**: User vector [1, 0, 1] (likes action, not romance, likes sci-fi) × Item vector [1, 1, 0] (action, romance, not sci-fi)
- **Result**: Outer product shows interaction strength for each feature pair
- **Why useful**: Captures that "action-loving users" might like "action movies" more than "romance movies"

**2. Neural Tensor Networks (Knowledge Graphs):**
- **Problem**: How do entities relate to each other?
- **Solution**: Use outer product to model entity-relation-entity triplets
- **Example**: "Paris" × "capital_of" × "France" - the outer product captures the relationship strength
- **Why useful**: Different relations (capital_of vs. located_in) have different interaction patterns

**3. Bilinear Pooling (Computer Vision):**
- **Problem**: How do different spatial locations interact in images?
- **Solution**: Outer product of spatial features captures location interactions
- **Example**: Features from top-left corner × features from bottom-right corner
- **Why useful**: Captures spatial relationships like "object A is above object B"

**4. Self-Attention in Transformers:**
- **Problem**: How much should each word attend to every other word?
- **Solution**: $QK^T$ computes attention scores between all word pairs
- **Example**: Query "cat" × Key "animal" = high attention (related)
- **Why useful**: Captures long-range dependencies in text

**5. Covariance Matrix (PCA):**
- **Problem**: How do features vary together?
- **Solution**: $\frac{1}{n}XX^T$ computes feature covariance
- **Example**: Height and weight features - outer product shows how they co-vary
- **Why useful**: Principal components are directions of maximum covariance

**6. Word Co-occurrence (NLP):**
- **Problem**: Which words appear together frequently?
- **Solution**: Outer product of word vectors captures co-occurrence patterns
- **Example**: "cat" and "dog" might have high interaction in the outer product
- **Why useful**: Captures semantic relationships between words

**Key Insight**: The outer product is powerful because it explicitly models pairwise interactions that simple linear models miss. It's like asking "what happens when feature A meets feature B?" for every possible pair.

Question 3. [E] What does it mean for two vectors to be linearly independent?

**Solution:** Linear independence is crucial in ML for understanding feature relationships and model capacity:

**Intuitive Understanding:**
Think of linear independence like having "unique information sources." Two vectors are linearly independent if each provides information that the other cannot provide. It's like having two different witnesses to an event - each has a unique perspective.

**Visual Analogy:**
- **Independent vectors**: Like two perpendicular arrows - you can't make one from the other
- **Dependent vectors**: Like two parallel arrows - one is just a scaled version of the other
- **In 2D**: Two vectors are independent if they're not on the same line
- **In 3D**: Three vectors are independent if they don't all lie in the same plane

**Mathematical Definition:**
Two vectors $\vec{v_1}$ and $\vec{v_2}$ are linearly independent if:
- **No non-trivial combination equals zero**: $c_1\vec{v_1} + c_2\vec{v_2} = \vec{0}$ only when $c_1 = c_2 = 0$
- **Practical test**: One vector is not a scalar multiple of the other
- **Geometric test**: They don't point in the same (or opposite) direction

**Step-by-Step Examples:**

**Example 1: Independent Vectors**
- $\vec{v_1} = [1, 0]$ and $\vec{v_2} = [0, 1]$
- **Test**: Can we make $[0, 1]$ from $[1, 0]$? No! (No scalar multiple works)
- **Result**: Independent - each provides unique directional information

**Example 2: Dependent Vectors**
- $\vec{v_1} = [1, 2]$ and $\vec{v_2} = [2, 4]$
- **Test**: Can we make $[2, 4]$ from $[1, 2]$? Yes! $2 \times [1, 2] = [2, 4]$
- **Result**: Dependent - $\vec{v_2}$ provides no new information

**ML Applications Explained:**

**1. Feature Selection:**
- **Problem**: Do we have redundant features?
- **Solution**: Remove linearly dependent features
- **Example**: If we have "height in cm" and "height in inches", we only need one
- **Why important**: Redundant features waste computational resources and can cause overfitting

**2. Dimensionality Reduction (PCA):**
- **Problem**: How many independent directions capture the data?
- **Solution**: PCA finds linearly independent directions of maximum variance
- **Example**: If all data points lie on a line, we only need 1 dimension, not 2
- **Why important**: Reduces complexity while preserving essential information

**3. Overfitting Prevention:**
- **Problem**: Too many similar features can cause overfitting
- **Solution**: Ensure features are linearly independent
- **Example**: Having both "age" and "age_squared" might be redundant if they're highly correlated
- **Why important**: Independent features provide unique information for learning

**4. Model Interpretability:**
- **Problem**: Which features actually matter?
- **Solution**: Linearly independent features have clearer individual contributions
- **Example**: If "income" and "salary" are dependent, we can't tell which one matters
- **Why important**: Helps understand what the model is actually learning

**Computational Considerations:**

**Rank of Feature Matrix:**
- **What it means**: Number of linearly independent features
- **How to compute**: Use SVD or QR decomposition
- **ML insight**: This tells us the true dimensionality of our feature space

**Condition Number:**
- **What it means**: How close to linear dependence our features are
- **High condition number**: Features are nearly dependent (bad for optimization)
- **Low condition number**: Features are well-separated (good for optimization)

**Real-World Examples:**
- **Independent**: Age and income (generally uncorrelated)
- **Dependent**: Height in cm and height in inches (perfectly correlated)
- **Near-dependent**: Temperature in Celsius and Fahrenheit (highly correlated but not perfectly)

Question 4. [M] Given two sets of vectors $A = a_1, a_2, a_3, \cdots , a_n$ and $B = b_1, b_2, b_3, \cdots , b_m$. How do you check that they share the same basis?

**Solution:** This problem is fundamental in ML for comparing feature spaces and understanding model representations:

**Intuitive Understanding:**
Think of this like asking "Do these two sets of building blocks allow us to build the same structures?" If set A can build everything that set B can build, and vice versa, then they're equivalent. In ML, this means "Do these two feature sets capture the same information?"

**ML Context Explained:**

**1. Feature Space Comparison:**
- **Problem**: Do different feature engineering approaches give us the same information?
- **Example**: Does using TF-IDF vs. word embeddings capture the same text information?
- **Why important**: Helps choose the best feature engineering approach

**2. Model Interpretability:**
- **Problem**: Do different neural network layers learn equivalent representations?
- **Example**: Does layer 3 of a CNN capture the same visual features as layer 5?
- **Why important**: Understanding what each layer learns

**3. Transfer Learning:**
- **Problem**: Are pre-trained features sufficient for a new task?
- **Example**: Can ImageNet features work for medical image classification?
- **Why important**: Determines if we need to retrain or can reuse features

**Step-by-Step Methods:**

**Method 1: Rank Comparison (Most Efficient)**
- **Step 1**: Compute rank(A), rank(B), and rank([A, B])
- **Step 2**: Check if rank([A, B]) = rank(A) = rank(B)
- **Step 3**: If true, then span(A) = span(B)
- **Intuition**: If combining A and B doesn't increase the rank, then B doesn't add new information beyond A
- **Example**: If rank(A) = 3, rank(B) = 3, rank([A,B]) = 3, then A and B span the same 3D space

**Method 2: Projection Test**
- **Step 1**: For each vector $a_i$ in A, project it onto span(B)
- **Step 2**: Check if the projection equals the original vector: $\text{proj}_{span(B)}(a_i) = a_i$
- **Step 3**: Repeat for each vector $b_j$ in B, projecting onto span(A)
- **Step 4**: If all projections equal originals, then span(A) = span(B)
- **Intuition**: If every vector in A can be perfectly reconstructed from B, then A doesn't add new information

**Method 3: SVD-Based Approach**
- **Step 1**: Compute SVD of the combined matrix [A | B]
- **Step 2**: Look at the singular values - they indicate the effective dimensionality
- **Step 3**: Compare with SVD of A alone and B alone
- **Step 4**: If the effective dimensions are the same, they span the same space
- **Why better**: Handles numerical precision issues common in real datasets

**Detailed Example:**

**Given:**
- Set A: $a_1 = [1, 0, 0]$, $a_2 = [0, 1, 0]$
- Set B: $b_1 = [1, 1, 0]$, $b_2 = [1, -1, 0]$

**Method 1 - Rank Comparison:**
- rank(A) = 2 (two independent vectors)
- rank(B) = 2 (two independent vectors)
- rank([A, B]) = 2 (no new information when combined)
- **Result**: They span the same 2D space

**Method 2 - Projection Test:**
- Project $a_1 = [1, 0, 0]$ onto span(B): Since $[1, 0, 0] = \frac{1}{2}[1, 1, 0] + \frac{1}{2}[1, -1, 0]$, the projection equals the original
- **Result**: A can be perfectly reconstructed from B

**Computational Considerations for ML:**

**Numerical Stability:**
- **Problem**: Large, ill-conditioned matrices cause numerical errors
- **Solution**: Use SVD instead of Gaussian elimination
- **Why**: SVD is more numerically stable for real-world data

**Efficiency:**
- **Problem**: High-dimensional data is computationally expensive
- **Solution**: Use randomized SVD or iterative methods
- **Why**: Approximate methods are often sufficient for ML applications

**Regularization:**
- **Problem**: Near-singular matrices cause instability
- **Solution**: Add small regularization terms (e.g., $\lambda I$)
- **Why**: Prevents numerical issues while preserving the essential structure

**Real-World ML Examples:**

**1. Word Embeddings Comparison:**
- **Question**: Do Word2Vec and GloVe capture the same semantic information?
- **Method**: Compare the spans of their embedding spaces
- **Result**: Often they span similar but not identical spaces

**2. CNN Layer Analysis:**
- **Question**: Do different CNN layers learn equivalent visual features?
- **Method**: Compare the feature spaces learned by different layers
- **Result**: Early layers capture low-level features, later layers capture high-level features

**3. Transfer Learning:**
- **Question**: Can ImageNet features work for medical images?
- **Method**: Check if medical image features lie in the span of ImageNet features
- **Result**: Often they don't, requiring fine-tuning or retraining

Question 5. [M] Given $n$ vectors, each of $d$ dimensions. What is the dimension of their span?

**Solution:** Span dimension is crucial in ML for understanding model capacity and feature space complexity:

**Intuitive Understanding:**
Think of span dimension as "how many independent directions can we explore with these vectors?" It's like asking "how many different ways can we combine these building blocks to create new things?" The span dimension tells us the true complexity of our feature space.

**ML Context Explained:**

**1. Model Capacity:**
- **Intuition**: Higher span dimension = more ways to represent data = more expressive model
- **Example**: A model with span dimension 10 can represent more complex patterns than one with span dimension 5
- **Trade-off**: More capacity can lead to overfitting if we don't have enough data

**2. Feature Engineering:**
- **Problem**: How many independent features do we actually have?
- **Example**: If we have 100 features but span dimension is only 20, we have 80 redundant features
- **Solution**: Remove redundant features to reduce complexity

**3. Representation Learning:**
- **Problem**: What's the intrinsic dimensionality of learned representations?
- **Example**: Autoencoders learn to compress data into lower-dimensional representations
- **Insight**: The span dimension tells us how much information we can actually capture

**Step-by-Step Calculation:**

**Given**: $n$ vectors, each of $d$ dimensions

**Step 1: Understand the constraints**
- **Maximum possible dimension**: $\min(n, d)$
  - Cannot exceed the number of vectors ($n$) - we can't create more dimensions than vectors we have
  - Cannot exceed the dimension of the ambient space ($d$) - we can't exceed the space we're working in

**Step 2: Find the actual dimension**
- **Method 1**: Use rank of the matrix formed by the vectors
- **Method 2**: Use SVD and count non-zero singular values
- **Method 3**: Use QR decomposition and count non-zero diagonal elements

**Concrete Examples:**

**Example 1: Simple Case**
- **Given**: 3 vectors in 2D space: $[1,0]$, $[0,1]$, $[1,1]$
- **Maximum possible**: $\min(3, 2) = 2$
- **Actual dimension**: 2 (all vectors are in the 2D plane)
- **ML insight**: We have 3 features but only 2 are independent

**Example 2: Redundant Features**
- **Given**: 4 vectors in 3D space: $[1,0,0]$, $[0,1,0]$, $[2,0,0]$, $[0,2,0]$
- **Maximum possible**: $\min(4, 3) = 3$
- **Actual dimension**: 2 (third and fourth vectors are just scaled versions of first two)
- **ML insight**: We have 4 features but only 2 are independent

**Example 3: High-Dimensional Data**
- **Given**: 1000 vectors in 10000D space
- **Maximum possible**: $\min(1000, 10000) = 1000$
- **Actual dimension**: Often much less (e.g., 50-200)
- **ML insight**: Real data often has much lower intrinsic dimension than ambient dimension

**ML Applications Explained:**

**1. Feature Space Analysis:**
- **Problem**: How many independent features do we actually have?
- **Solution**: Compute span dimension of feature matrix
- **Example**: If we have 1000 features but span dimension is 100, we have 900 redundant features
- **Action**: Remove redundant features to reduce complexity and prevent overfitting

**2. PCA (Principal Component Analysis):**
- **Problem**: How many principal components should we keep?
- **Solution**: Number of significant principal components = span dimension of centered data
- **Example**: If span dimension is 50, keep only the first 50 principal components
- **Why**: Components beyond the span dimension capture only noise

**3. Neural Network Analysis:**
- **Layer Capacity**: Each layer's span dimension determines its representational power
- **Bottleneck Layers**: Span dimension < input dimension creates compression
- **Example**: Autoencoder with 1000D input → 100D hidden layer → 1000D output
- **Insight**: The 100D hidden layer has span dimension ≤ 100, creating a bottleneck

**4. Representation Learning:**
- **Problem**: What's the intrinsic dimensionality of learned representations?
- **Solution**: Compute span dimension of learned features
- **Example**: Word embeddings might be 300D but span only 200 dimensions
- **Insight**: We can compress them to 200D without losing information

**Computational Methods:**

**Method 1: SVD (Singular Value Decomposition)**
- **Step 1**: Compute SVD of the matrix $X = [v_1, v_2, \ldots, v_n]$
- **Step 2**: Count non-zero singular values
- **Result**: $\text{rank}(X) = \text{number of non-zero singular values}$
- **Why good**: Handles numerical precision issues

**Method 2: QR Decomposition**
- **Step 1**: Compute QR decomposition of the matrix
- **Step 2**: Count non-zero diagonal elements in R
- **Result**: Number of non-zero diagonal elements = rank
- **Why good**: More efficient than SVD for some cases

**Method 3: Eigenvalue Analysis**
- **Step 1**: Compute covariance matrix $C = XX^T$
- **Step 2**: Count non-zero eigenvalues
- **Result**: Number of non-zero eigenvalues = rank
- **Why good**: Useful for understanding variance structure

**Real-World Examples:**

**1. Image Data:**
- **Problem**: 1000×1000 pixel images = 1,000,000 dimensions
- **Reality**: Span dimension might be only 100-1000
- **Insight**: Most images can be represented in much lower dimensions
- **Application**: Image compression, dimensionality reduction

**2. Text Embeddings:**
- **Problem**: 300D word vectors for vocabulary of 100,000 words
- **Reality**: Span dimension might be only 50-200
- **Insight**: Word relationships can be captured in much lower dimensions
- **Application**: Efficient storage, faster computation

**3. Recommendation Systems:**
- **Problem**: User-item matrices with millions of users and items
- **Reality**: Span dimension is typically much lower (e.g., 50-500)
- **Insight**: User preferences can be captured in low-dimensional space
- **Application**: Matrix factorization, collaborative filtering

**Practical Computation:**
$$\text{dim}(\text{span}\{v_1, v_2, \ldots, v_n\}) = \text{rank}([v_1, v_2, \ldots, v_n])$$

**Key Insight**: The span dimension tells us the true complexity of our data. It's often much lower than the ambient dimension, which is why dimensionality reduction and compression work so well in ML.

6. Norms and metrics
   
Question i. [E] What's a norm? What are $L_0, L_1, L_2, L_{\infty}$ norms?

**Solution:**

**What is a norm?**
A norm measures vector "size" and is fundamental in ML for regularization, optimization, and distance calculations:

**Intuitive Understanding:**
Think of a norm as a "ruler" that measures how "big" or "important" a vector is. Different norms give different answers to the question "how big is this vector?" depending on what we care about.

**Mathematical Properties:**
1. **Positive definiteness**: $||v|| \geq 0$ and $||v|| = 0$ iff $v = 0$ (size is never negative, zero only for zero vector)
2. **Homogeneity**: $||\alpha v|| = |\alpha| \cdot ||v||$ (scaling the vector scales the norm proportionally)
3. **Triangle inequality**: $||u + v|| \leq ||u|| + ||v||$ (shortest path between two points is a straight line)

**Detailed ML Applications:**

**$L_0$ norm (Sparsity)**: $||x||_0 = \text{number of non-zero elements}$
- **Intuition**: "How many features are actually being used?"
- **ML Application**: Feature selection - minimize $L_0$ to select most important features
- **Example**: If $x = [0, 5, 0, 3, 0]$, then $||x||_0 = 2$ (only 2 features are active)
- **Problem**: Not differentiable, so we use $L_1$ as a convex relaxation
- **Why important**: Sparse models are easier to interpret and faster to compute

**$L_1$ norm (Lasso regularization)**: $`||x||_1 = \sum_{i=1}^n |x_i|`$
- **Intuition**: "What's the total absolute value of all features?"
- **ML Application**: L1 regularization $\min \text{loss} + \lambda ||w||_1$ encourages sparsity
- **Example**: If $x = [2, -3, 0, 1]$, then $||x||_1 = 2 + 3 + 0 + 1 = 6$
- **Why sparsity**: L1 penalty drives some weights to exactly zero
- **Robust to outliers**: Less sensitive than $L_2$ to extreme values
- **Feature selection**: Automatically selects important features by zeroing others

**$L_2$ norm (Ridge regularization)**: $`||x||_2 = \sqrt{\sum_{i=1}^n x_i^2}`$
- **Intuition**: "What's the Euclidean distance from origin?" (Pythagorean theorem)
- **ML Application**: L2 regularization $\min \text{loss} + \lambda ||w||_2^2$ prevents overfitting
- **Example**: If $x = [3, 4]$, then $||x||_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = 5$
- **Why good for optimization**: L2 gradients are well-behaved (smooth)
- **Distance metrics**: Euclidean distance for clustering, similarity
- **Geometric interpretation**: Length of the vector in Euclidean space

**$L_{\infty}$ norm (Max pooling)**: $`||x||_{\infty} = \max_{i} |x_i|`$
- **Intuition**: "What's the largest absolute value among all features?"
- **ML Application**: Max pooling in CNNs captures strongest activation
- **Example**: If $x = [2, -5, 1, 3]$, then $||x||_{\infty} = \max(2, 5, 1, 3) = 5$
- **Robust optimization**: Minimize worst-case loss (focus on the biggest error)
- **Normalization**: Used in batch normalization for stability
- **Why useful**: Focuses on the most important feature (the maximum)

**Step-by-Step Examples:**

**Example 1: Vector $x = [3, -4, 0, 2]$**
- **$L_0$**: $||x||_0 = 3$ (3 non-zero elements)
- **$L_1$**: $||x||_1 = 3 + 4 + 0 + 2 = 9$ (sum of absolute values)
- **$L_2$**: $||x||_2 = \sqrt{3^2 + 4^2 + 0^2 + 2^2} = \sqrt{9 + 16 + 0 + 4} = \sqrt{29} \approx 5.39$
- **$L_{\infty}$**: $||x||_{\infty} = \max(3, 4, 0, 2) = 4$ (maximum absolute value)

**Example 2: Sparse vector $x = [0, 0, 5, 0, 0]$**
- **$L_0$**: $||x||_0 = 1$ (only 1 non-zero element)
- **$L_1$**: $||x||_1 = 5$ (sum of absolute values)
- **$L_2$**: $||x||_2 = 5$ (Euclidean distance)
- **$L_{\infty}$**: $||x||_{\infty} = 5$ (maximum absolute value)

**ML Intuition:**
- **$L_0$**: "How many features matter?" (sparsity)
- **$L_1$**: "What's the total impact?" (robust to outliers)
- **$L_2$**: "What's the geometric distance?" (smooth optimization)
- **$L_{\infty}$**: "What's the worst case?" (focus on extremes)

Question ii. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

**Solution:**

**Intuitive Understanding:**
Think of norms and metrics as different ways to measure "size" and "distance":
- **Norms**: Measure the "size" of a single vector (like measuring the length of a stick)
- **Metrics**: Measure the "distance" between two points (like measuring the distance between two cities)
- **Key insight**: Every norm can create a metric, but not every metric can create a norm

**ML Context - Norms vs Metrics:**

**Norms in ML:**
- **Purpose**: Measure vector "size" for regularization, optimization, feature scaling
- **Examples**: L1/L2 regularization, gradient clipping, feature normalization
- **Requirement**: Need vector space structure (addition and scalar multiplication)

**Metrics in ML:**
- **Purpose**: Measure "distance" for clustering, similarity, loss functions
- **Examples**: Euclidean distance for k-NN, cosine similarity for text, Hamming distance for categorical data
- **Requirement**: Work on any data (vectors, strings, graphs, etc.)

**Step-by-Step: From Norm to Metric (Always Possible)**

**Step 1: Start with any norm**
- Example: $L_2$ norm $||x||_2 = \sqrt{\sum x_i^2}$

**Step 2: Create distance function**
- Formula: $d(x,y) = ||x - y||$
- Intuition: Distance = size of the difference vector

**Step 3: Verify it's a metric**
- **Positive definiteness**: $d(x,y) \geq 0$ and $d(x,y) = 0$ iff $x = y$ ✓
- **Symmetry**: $d(x,y) = d(y,x)$ ✓
- **Triangle inequality**: $d(x,z) \leq d(x,y) + d(y,z)$ ✓

**Concrete Examples:**

**Example 1: $L_2$ norm → Euclidean distance**
- **Norm**: $||x||_2 = \sqrt{\sum x_i^2}$
- **Metric**: $d(x,y) = ||x - y||_2 = \sqrt{\sum (x_i - y_i)^2}$
- **ML use**: k-NN, clustering, similarity search

**Example 2: $L_1$ norm → Manhattan distance**
- **Norm**: $||x||_1 = \sum |x_i|$
- **Metric**: $d(x,y) = ||x - y||_1 = \sum |x_i - y_i|$
- **ML use**: Robust to outliers, sparse data

**Example 3: $L_{\infty}$ norm → Chebyshev distance**
- **Norm**: $||x||_{\infty} = \max_i |x_i|$
- **Metric**: $d(x,y) = ||x - y||_{\infty} = \max_i |x_i - y_i|$
- **ML use**: Max pooling, robust optimization

**Step-by-Step: From Metric to Norm (Not Always Possible)**

**Step 1: Start with a metric**
- Example: Cosine distance $d(x,y) = 1 - \frac{x \cdot y}{||x||_2 ||y||_2}$

**Step 2: Check required conditions**
- **Translation invariance**: $d(x + z, y + z) = d(x, y)$?
- **Homogeneity**: $d(\alpha x, \alpha y) = |\alpha| d(x, y)$?
- **Origin condition**: $d(0, x) = ||x||$?

**Step 3: Determine if possible**
- If all conditions hold → can define norm
- If any condition fails → cannot define norm

**Detailed Examples:**

**Example 1: Euclidean metric → $L_2$ norm ✓**
- **Metric**: $d(x,y) = \sqrt{\sum (x_i - y_i)^2}$
- **Translation invariance**: $d(x + z, y + z) = \sqrt{\sum ((x_i + z_i) - (y_i + z_i))^2} = \sqrt{\sum (x_i - y_i)^2} = d(x,y)$ ✓
- **Homogeneity**: $d(\alpha x, \alpha y) = \sqrt{\sum (\alpha x_i - \alpha y_i)^2} = |\alpha| \sqrt{\sum (x_i - y_i)^2} = |\alpha| d(x,y)$ ✓
- **Origin condition**: $d(0, x) = \sqrt{\sum (0 - x_i)^2} = \sqrt{\sum x_i^2} = ||x||_2$ ✓
- **Result**: Can define $L_2$ norm

**Example 2: Cosine distance → Cannot define norm ✗**
- **Metric**: $d(x,y) = 1 - \frac{x \cdot y}{||x||_2 ||y||_2}$
- **Translation invariance test**: $d(x + z, y + z) = 1 - \frac{(x + z) \cdot (y + z)}{||x + z||_2 ||y + z||_2}$
- **Problem**: This doesn't equal $d(x,y)$ because the dot product and norms change
- **Result**: Cannot define norm (not translation invariant)

**Example 3: Manhattan metric → $L_1$ norm ✓**
- **Metric**: $d(x,y) = \sum |x_i - y_i|$
- **Translation invariance**: $d(x + z, y + z) = \sum |(x_i + z_i) - (y_i + z_i)| = \sum |x_i - y_i| = d(x,y)$ ✓
- **Homogeneity**: $d(\alpha x, \alpha y) = \sum |\alpha x_i - \alpha y_i| = |\alpha| \sum |x_i - y_i| = |\alpha| d(x,y)$ ✓
- **Origin condition**: $d(0, x) = \sum |0 - x_i| = \sum |x_i| = ||x||_1$ ✓
- **Result**: Can define $L_1$ norm

**Practical ML Considerations:**

**Feature Scaling:**
- **Problem**: Different features have different scales
- **Solution**: Use norms to normalize features before applying distance-based algorithms
- **Example**: Normalize features to unit $L_2$ norm before k-NN

**Regularization:**
- **Problem**: Model overfitting
- **Solution**: Use norms in loss functions to control model complexity
- **Example**: L1 regularization encourages sparsity, L2 regularization prevents large weights

**Optimization:**
- **Problem**: Different norms have different optimization landscapes
- **L1**: Non-differentiable at zero, encourages sparsity
- **L2**: Smooth everywhere, good for gradient descent
- **L∞**: Focuses on worst-case errors

**Key Insight**: Norms and metrics are closely related but serve different purposes. Every norm creates a metric, but only metrics with vector space properties can create norms. This distinction is crucial for choosing the right distance function in ML applications.
