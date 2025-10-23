# Matrices

1. [E] Why do we say that matrices are linear transformations?

2. [E] What's the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?

3. [E] What does the determinant of a matrix represent?

4. [E] What happens to the determinant of a matrix if we multiply one of its rows by a scalar $t \in \mathbb{R}$?

5. [M] A $4 \times 4$ matrix has four eigenvalues $3, 3, 2, -1$. What can we say about the trace and the determinant of this matrix?

6. [M] Given the following matrix:
   $$
   \begin{bmatrix}
   1 & 4 & -2 \\
   -1 & 3 & 2 \\
   3 & 5 & -6
   \end{bmatrix}
   $$
   Without explicitly using the equation for calculating determinants, what can we say about this matrix's determinant?
   **Hint:** Rely on a property of this matrix to determine its determinant.

7. [M] What's the difference between the covariance matrix $A^T A$ and the Gram matrix $A A^T$?

8. Given $A \in \mathbb{R}^{n \times m}$ and $b \in \mathbb{R}^n$
   i. [M] Find $x$ such that: $Ax = b$.
   ii. [E] When does this have a unique solution?
   iii. [M] Why is it that when $A$ has more columns than rows, $Ax = b$ has multiple solutions?
   iv. [M] Given a matrix $A$ with no inverse. How would you solve the equation $Ax = b$? What is the pseudoinverse and how to calculate it?

9. Derivatives are the backbone of gradient descent.
   i. [E] What does a derivative represent?
   ii. [M] What's the difference between a derivative, gradient, and Jacobian?

10. [H] Say we have the weights $w \in \mathbb{R}^{d \times m}$ and a mini-batch $x$ of $n$ elements, each element is of the shape $1 \times d$ so that $x \in \mathbb{R}^{n \times d}$. We have the output $y = f(x; w) = xw$. What is the dimension of the Jacobian $\frac{\partial y}{\partial x}$?

11. [H] Given a very large symmetric matrix $A$ that doesn't fit in memory, say $A \in \mathbb{R}^{1M \times 1M}$ and a function $f$ that can quickly compute $f(x) = Ax$ for $x \in \mathbb{R}^{1M}$. Find the unit vector $x$ such that $x^T Ax$ is minimal.
    
    **Hint:** Can you frame it as an optimization problem and use gradient descent to find an approximate solution?