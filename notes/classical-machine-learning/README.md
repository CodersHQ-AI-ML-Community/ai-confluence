# Classical ML roadmap

## Textbooks
- [Machine Learning by Tom Mitchell](https://www.cs.cmu.edu/~tom/mlbook.html)
- [(PRML) Pattern Recognition and Machine Learning by Chris Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

## Roadmap
All chapters reference Mitchell's textbook until specified otherwise.

### An introduction
- Introduction - Chapter 1
	- understand the goals of machine learning
- Concept Learning - Chapter 2
	- What is concept learning? What are the goals of function approximation?
	- What are inductive biases?

### Supervised Learning
- Decision Trees - Chapter 3
	- Understand the ID3 algorithm
	- Occam's Razor - Why Prefer Short Hypotheses?
	- Issuew with Decision Tree Learning
	
- Neural nets - Chapter 4
	- Understand the perceptron
		- Why Perceptron's fail for non-linear classifiers?
	- Understand how backpropagation works
		- Error function derivation
		- Understand the bias-variance trade-off
	- Alternative error functions
	
- K-Nearest neighbors - Chapter 8
	- Understand the inductive biases and workings of K-nearest neighbors
	- Lazy vs Eager learners
	
- Support Vector Machines - Chapter 7 (PRML)
	- Good 3 part series by [StatQuest](https://www.youtube.com/watch?v=efR1C6CvhmE&pp=ygUOc3RhdHF1ZXN0IFNWTXM%3D)
	- What is the optimization problem here?
	- Difference in compute times based on kernel used - polynomials vs rbf kernels.

- Boosting - Chapter 14.3 (PRML)
	- Understand the Adaboost classifier.
		- How an ensemble of weak learners (decision stumps) can give really good results?
		- How the misclassified training examples are being sampled which is focused on by the next weak learner?
	- Good explanation by [StatQuest](https://www.youtube.com/watch?v=LsK-xG1cLYA&pp=ygUTYWRhYm9vc3QgY2xhc3NpZmllcg%3D%3D)
	
## Unsupervised Learning
- Clustering
	- K-means
	- Gaussian Mixture Models - Expectation Maximization

- Dimension Reduction
	- Principal Component Analysis (PCA)
	- Independent Component Analysis (ICA)
	- Feature Selection
		- Filtering
			- How Decision Trees can help find feature importances?
		- Wrapping
			- Forward Elimination
			- Backward Elimination
	- [Manifold Learning](https://sites.gatech.edu/omscs7641/2024/03/10/no-straight-lines-here-the-wacky-world-of-non-linear-manifold-learning/)
		- Sammon Mapping
		- Isomaps
		- Multidimensional Scaling (MDS)
		- t-distributed Stochastic Neighbor Embeddings (t-SNE)
		
		
## Reinforcement Learning
- Value Iteration
- Policy Iteration
- Q-learning
	
