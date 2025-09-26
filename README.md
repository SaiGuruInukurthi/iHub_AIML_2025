# iHub AIML 2025 - Machine Learning Notebooks

This repository contains comprehensive machine learning notebooks covering various topics from basic concepts to advanced algorithms. Each module includes hands-on implementations, visualizations, and practical exercises.

## Repository Structure

```
├── Mod0_Linear_Algebra.ipynb                   # Linear algebra fundamentals
├── Mod0_Probability_Basics.ipynb               # Probability theory basics
├── Mod1_Lab1_Features.ipynb                    # Feature engineering basics
├── Mod1_Lab2_ML_terms_and_metrics.ipynb        # ML terminology and evaluation metrics
├── Mod1_Lab3_Data_Augmentation.ipynb           # Data augmentation techniques
├── Mod1_Lab4_Transforming_data_using_linear_algebra.ipynb  # Linear algebra for ML
├── Mod2_Lab1_Basic_Plots.ipynb                 # Data visualization basics
├── Mod2_Lab2_Manifold_Learning_Methods.ipynb   # Manifold learning techniques
├── Mod2_Lab3_Principal_Components_Analysis_(PCA).ipynb     # PCA implementation
├── Mod2_Project.ipynb                          # SARS-CoV-2 genomic analysis project (ENHANCED)
├── Mod3_Lab1_Understanding_Distance_metrics_and_Introduction_to_KNN.ipynb  # Distance metrics & KNN
├── Mod3_Lab2_Using_KNN_for_Text_Classification.ipynb      # KNN for text analysis
├── Mod3_Lab3_Implementing_KNN_from_scratch_and_visualize_Algorithm_performance.ipynb  # KNN from scratch
├── Mod3_project.ipynb                          # Diabetes prediction project with comprehensive KNN analysis
├── Mod4_Lab1_Perceptron_and_Gradient_Descent.ipynb        # Perceptron & gradient descent
├── Mod4_Lab2_Introduction_to_Gradient_Descent.ipynb       # Advanced gradient descent
├── Mod4_Lab3_Gradient_Descent.ipynb            # Advanced gradient descent techniques
├── Mod5_Lab1_Linear_Regression_MSE_and_Polynomial_Regression.ipynb  # Linear regression & polynomial regression
├── Mod5_Lab2_Loss_Functions.ipynb              # Loss functions analysis
├── Mod5_Lab3_Clustering.ipynb                  # Clustering algorithms (K-Means, Hierarchical, DBSCAN)
├── Mod6_Lab1_Implementing_forward_propagation_and_back_propagation.ipynb  # Neural networks fundamentals
├── Mod6_Lab2_Training_a_Neural_Network.ipynb   # Neural network training from scratch
├── Mod6_Lab3_CNN_&_Architectures.ipynb         # CNN implementation, visualization & transfer learning
├── car_evaluation.csv                          # Car evaluation dataset
├── diabetes.csv                                # Pima Indian Diabetes dataset (for Mod3_project)
├── INDIA_685.csv                               # India dataset
├── Mall_Customers.csv                          # Mall customers dataset for clustering
├── Wholesale customers data.csv                # Wholesale customers dataset
├── reviews.csv                                 # Reviews dataset
├── sequences.fasta                             # Biological sequences
└── spam.csv                                    # Spam detection dataset
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/SaiGuruInukurthi/iHub_AIML_2025.git
cd iHub_AIML_2025
```

## Setting Up Python Environment

### Option A: Single Virtual Environment (Recommended for beginners)

Create one virtual environment for all notebooks:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### Option B: Module-Specific Virtual Environments (Advanced)

For different modules that may have conflicting dependencies:

```bash
# For Module 0 notebooks (Mathematical Foundations)
python -m venv .venv_mod0
.venv_mod0\Scripts\activate  # Windows
pip install numpy matplotlib scipy scikit-image sympy seaborn plotly

# For Module 0 Probability Basics (separate environment if needed)
python -m venv .venv_mod0_probability
.venv_mod0_probability\Scripts\activate  # Windows
pip install numpy matplotlib seaborn scipy pandas plotly sympy statsmodels

# For Module 1 notebooks
python -m venv .venv_mod1
.venv_mod1\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn plotly nltk wikipedia keras tensorflow

# For Module 2 notebooks
python -m venv .venv_mod2
.venv_mod2\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn plotly

# For Module 3 notebooks
python -m venv .venv_mod3
.venv_mod3\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn scipy nltk

# For Module 3 Project (Diabetes Prediction)
python -m venv mod3_project_venv
mod3_project_venv\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter notebook ipykernel

# For Module 4 notebooks
python -m venv .venv_mod4
.venv_mod4\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# For Module 5 notebooks
python -m venv .venv_mod5
.venv_mod5\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# For Module 6 notebooks (Neural Networks & CNNs)
python -m venv .venv_mod6
.venv_mod6\Scripts\activate  # Windows
pip install numpy matplotlib scikit-learn tensorflow keras torch torchvision opencv-python gdown pillow
```

## Required Dependencies

Create a `requirements.txt` file or install these packages individually:

```txt
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.10.0
plotly>=5.14.0
nltk>=3.8.0
jupyter>=1.0.0
ipykernel>=6.20.0
biopython>=1.81.0  # For sequences.fasta file
scikit-image>=0.20.0  # For SVD image processing (Module 0)
sympy>=1.12.0  # Symbolic mathematics (Module 0)
statsmodels>=0.14.0  # Statistical modeling (Module 0 Probability)
tensorflow>=2.12.0  # For deep learning (Module 6)
keras>=2.12.0       # For neural networks
torch>=2.0.0        # PyTorch for deep learning (Module 6)
torchvision>=0.15.0 # Computer vision with PyTorch (Module 6 CNNs)
opencv-python>=4.8.0  # Image processing (Module 6 CNNs)
gdown>=4.7.0        # Google Drive downloads (Module 6 datasets)
pillow>=9.5.0       # Image processing and PIL operations
```

## Running Notebooks

### Option 1: VS Code (Recommended)

#### Required Extensions:
1. **Python** (Microsoft) - Essential for Python support
2. **Jupyter** (Microsoft) - For notebook support
3. **Python Environment Manager** (Don Jayamanne) - For managing virtual environments
4. **Python Docstring Generator** (Nils Werner) - Optional but helpful

#### Setup Steps:
1. Open VS Code
2. Install the required extensions
3. Open the repository folder: `File > Open Folder > Select iHub_AIML_2025`
4. Select Python interpreter:
   - Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
   - Type "Python: Select Interpreter"
   - Choose your virtual environment's Python executable
5. Open any `.ipynb` file and start running cells

#### VS Code Tips:
- Use `Shift+Enter` to run current cell and move to next
- Use `Ctrl+Enter` to run current cell without moving
- Use the variable explorer to inspect variables
- Enable auto-save: `File > Auto Save`

### Option 2: JupyterLab

#### Installation:
```bash
# Activate your virtual environment first
pip install jupyterlab

# Install additional JupyterLab extensions (optional)
pip install jupyterlab-git  # Git integration
pip install jupyterlab_widgets  # Interactive widgets
```

#### Running JupyterLab:
```bash
# Navigate to repository directory
cd iHub_AIML_2025

# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Start JupyterLab
jupyter lab
```

#### JupyterLab Tips:
- Access at `http://localhost:8888`
- Use the file browser on the left to navigate
- Right-click notebooks for context menu options
- Use `Shift+Enter` to run cells

### Option 3: Jupyter Notebook (Classic)

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Install Jupyter Notebook
pip install notebook

# Start Jupyter Notebook
jupyter notebook
```

## Troubleshooting

### Common Issues:

#### 1. Module Not Found Errors
```bash
# Make sure virtual environment is activated
# Reinstall the missing package
pip install package_name
```

#### 2. Kernel Not Found in VS Code
- Open Command Palette (`Ctrl+Shift+P`)
- Type "Python: Select Interpreter"
- Choose the correct virtual environment

#### 3. Jupyter Kernel Issues
```bash
# Install ipykernel in your virtual environment
pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name=.venv --display-name="Python (iHub AIML)"
```

#### 4. Plot Not Displaying
```python
# Add this to notebook cells if plots don't show
%matplotlib inline
```

#### 5. File Not Found Errors
- Ensure you're running notebooks from the repository root directory
- Check that all CSV files are in the same directory as notebooks
- **diabetes.csv**: Required for Mod3_project.ipynb - download Pima Indian Diabetes dataset
- **car_evaluation.csv**: Required for Mod3_Lab3 - download from UCI repository

## Module-Specific Notes

### Module 0: Mathematical Foundations
- **Comprehensive Linear Algebra**: Matrix operations, eigendecomposition, SVD
- **Advanced Features**: System of equations solving, matrix power operations, norms
- **Practical Applications**: SVD image compression and reconstruction examples
- **Interactive Visualizations**: Image processing demonstrations with scikit-image
- **Probability Theory**: Comprehensive statistical foundations with real-world data analysis
- Requires: numpy, matplotlib, scipy, scikit-image, sympy, statsmodels, pandas, plotly
- **Status**: ✅ Both notebooks enhanced and fully functional

#### 🧮 Mod0_Linear_Algebra.ipynb - Complete Mathematical Foundation
This notebook provides comprehensive coverage of linear algebra concepts essential for ML:
- **Matrix Operations**: Transpose, dot product, matrix multiplication, outer products
- **Advanced Matrix Methods**: Multi-dot optimization, matrix powers, determinants, inverses
- **Eigenanalysis**: Eigenvalues, eigenvectors, and their practical interpretations
- **Linear Systems**: Solving systems of equations using NumPy's linear algebra solver
- **Matrix Decomposition**: Singular Value Decomposition (SVD) with practical applications
- **Image Processing**: SVD-based image compression and reconstruction demonstrations
- **Performance Analysis**: Comparing different matrix multiplication approaches
- **Comprehensive Examples**: From basic operations to advanced applications like dimensionality reduction

#### 📊 Mod0_Probability_Basics.ipynb - Statistical Foundations for ML
This notebook covers essential probability and statistics concepts with practical applications:
- **Basic Probability**: Fractions, combinatorics, sample spaces, and probability rules
- **Statistical Measures**: Mean, variance, standard deviation with real-world student grades data
- **Probability Distributions**: Normal, binomial, Poisson distributions with interactive visualizations
- **Data Analysis**: Real student grades dataset analysis with descriptive statistics
- **Interactive Plots**: Plotly visualizations for probability mass/density functions
- **Mathematical Foundations**: Using fractions module for exact probability calculations
- **Practical Examples**: Card drawing, dice rolling, and statistical sampling demonstrations
- **Environment**: Dedicated .venv_mod0_probability with comprehensive statistical packages

### Module 1: Foundations & Feature Engineering
- **Enhanced Feature Engineering**: Comprehensive text and image feature extraction
- **Multi-language Analysis**: N-gram analysis across English, French, Spanish, German
- **Advanced Explorations**: All lab questions answered with detailed code and analysis
- Requires: pandas, numpy, matplotlib, seaborn, wikipedia, nltk, keras (for MNIST), sklearn, plotly

#### 🔧 Mod1_Lab1_Features.ipynb - Complete Feature Engineering Guide
This notebook provides comprehensive coverage of feature extraction techniques:
- **Text Feature Engineering**: Character n-grams (1-5 grams) for language detection
- **Multi-language Analysis**: Wikipedia content analysis across 4+ languages
- **Image Feature Engineering**: MNIST digit analysis with 8+ feature types
- **Advanced Visualizations**: 2D heatmaps, 3D plots, parallel coordinates, PCA
- **Custom Feature Development**: Aspect ratio, symmetry, geometric features
- **Multi-class Analysis**: Feature performance across different digit combinations
- **Comprehensive Explorations**: All lab questions answered with working code
- **Datasets**: MNIST (auto-loaded), Wikipedia (auto-downloaded), synthetic text

### Module 2: Dimensionality Reduction & Bioinformatics
- Covers PCA, t-SNE, ISOMAP, and manifold learning
- **Enhanced Project**: SARS-CoV-2 genomic analysis with advanced dimensionality reduction
- Requires: scikit-learn, plotly, biopython, tqdm
- Datasets: INDIA_685.csv, sequences.fasta (685 SARS-CoV-2 genome sequences)

#### 🧬 Mod2_Project.ipynb - Enhanced Bioinformatics Analysis
This project has been significantly enhanced beyond basic requirements:
- **K-mer Analysis**: 7-mer profiling of 685 SARS-CoV-2 genome sequences
- **Mutation Profiling**: Detection and analysis of genomic mutations across Indian states
- **Advanced Dimensionality Reduction**: 
  - PCA with explained variance analysis
  - t-SNE with multiple perplexity optimization (5, 15, 30, 50)
  - ISOMAP with neighborhood analysis (5, 10, 15, 20 neighbors)
- **Comparative Analysis**: Side-by-side visualization of all three methods
- **Quantitative Metrics**: Silhouette analysis, Calinski-Harabasz scores, clustering evaluation
- **Geographic Clustering**: State-wise COVID-19 variant analysis and spread patterns

### Module 3: K-Nearest Neighbors
- Implementation from scratch and performance analysis
- **Enhanced Project**: Comprehensive diabetes prediction analysis with KNN
- Requires: scipy for distance metrics, nltk for text processing

#### 🩺 Mod3_project.ipynb - Diabetes Prediction Project ✅ COMPLETE
This comprehensive project analyzes diabetes prediction in Pima Indian Women using KNN classification:
- **Dataset**: diabetes.csv (768 records, 9 features) - Pima Indian Women diabetes study
- **Comprehensive EDA**: Correlation analysis, distribution plots, class balance analysis
- **Feature Analysis**: BMI, age, pregnancies, pedigree function relationships with diabetes
- **Scaling Comparison**: StandardScaler vs MinMaxScaler performance analysis
- **Visualization Suite**: 
  - Correlation heatmaps with seaborn
  - Pairplot analysis with diabetes outcome classification
  - Boxplot comparisons across all key features
  - Voronoi diagrams with PCA dimensionality reduction
  - Decision boundary visualization
- **KNN Optimization**: K-value selection using elbow method and cross-validation
- **Advanced Analysis**: 
  - K-Fold cross-validation with stratified sampling
  - Statistical significance testing with paired t-tests
  - Comprehensive performance metrics and error analysis
- **9 Complete Tasks**: All project tasks implemented with detailed analysis
- **Virtual Environment**: Dedicated mod3_project_venv for isolation
- **Dependencies**: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

### Module 4: Gradient Descent
- Advanced optimization techniques
- Requires: scipy for optimization functions

### Module 5: Linear Regression & Clustering
- Linear regression, polynomial regression, and loss functions
- Clustering algorithms: K-Means, Hierarchical, DBSCAN
- Requires: scikit-learn, matplotlib, seaborn for visualization

### Module 6: Neural Networks & Deep Learning
- Forward propagation and back propagation implementation
- Neural network training from scratch with custom implementations
- Convolutional Neural Networks (CNNs) and computer vision
- Transfer learning and fine-tuning pretrained models
- Requires: numpy, matplotlib, scikit-learn, torch, torchvision, opencv-python, gdown

#### 🧠 Mod6_Lab2_Training_a_Neural_Network.ipynb - Neural Network Tutorial
This notebook provides a comprehensive walkthrough of neural network concepts:
- **Theoretical Foundation**: Detailed explanations of neurons, layers, weights, biases, activation functions
- **Mathematical Background**: Weighted sums, sigmoid activation, cost functions, gradient descent
- **Custom Implementation**: Complete neural network class built from scratch
- **Practical Examples**: Training on Iris dataset with visualization of cost reduction
- **Learning Concepts**: Forward propagation, backpropagation, parameter optimization
- **Dependencies**: numpy, matplotlib, scikit-learn (Iris dataset)
- **Environment**: Compatible with mod6_lab2_venv or main .venv

#### 🖼️ Mod6_Lab3_CNN_&_Architectures.ipynb - Convolutional Neural Networks ✅ COMPLETE
This comprehensive notebook covers CNN fundamentals and advanced deep learning concepts:
- **Section 1: CNN Fundamentals**
  - Convolution operations with custom implementations
  - CNN implementation and training on MNIST dataset
  - CNN visualization and filter analysis
- **Section 2: Advanced CNN Concepts**
  - Effects of padding, kernel size, and stride
  - Pooling operations (Max pooling, Average pooling)
  - Transfer learning with ResNet18 on German Traffic Signs dataset
- **Key Features**:
  - Custom convolution functions for educational purposes
  - PyTorch CNN implementation with detailed explanations
  - Filter visualization and feature map analysis
  - Transfer learning comparison: Fine-tuning vs Feature extraction
  - Comprehensive exercise solutions covering transfer learning concepts
- **Datasets**: MNIST (auto-downloaded), German Traffic Signs (auto-downloaded), lotus.jpg (sample image)
- **Dependencies**: torch, torchvision, opencv-python, matplotlib, numpy, gdown
- **Environment**: Compatible with mod6_lab3_venv or main .venv

## Learning Path

1. **Start with Module 0**: 
   - `Mod0_Linear_Algebra.ipynb` - Essential mathematical foundations
   - `Mod0_Probability_Basics.ipynb` - Statistical concepts for ML
2. **Module 1**: Feature engineering and ML basics
3. **Module 2**: Data visualization and dimensionality reduction
4. **Module 3**: Classification algorithms (KNN)
5. **Module 4**: Optimization techniques
6. **Module 5**: Regression and clustering algorithms
7. **Module 6**: Neural networks and deep learning

## Contributing

If you find issues or want to improve the notebooks:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit: `git commit -m "Description of changes"`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## License

This project is for educational purposes. Please respect any dataset licenses and cite sources appropriately.

## Troubleshooting

### Common Issues

#### Module 6 (CNN & Deep Learning)
- **PyTorch Installation Issues**: Use `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` for CPU-only version
- **CUDA/GPU Issues**: Install appropriate CUDA version or use CPU-only PyTorch
- **German Traffic Signs Dataset Download**: Ensure stable internet connection, retry if download fails
- **OpenCV Installation**: Try `pip install opencv-python-headless` if regular opencv-python fails
- **Memory Issues**: Reduce batch size or use smaller model if running out of memory

#### General Issues
- **Virtual Environment**: Always activate your virtual environment before installing packages
- **Package Conflicts**: Create separate environments for different modules if needed
- **Jupyter Kernel**: Run `python -m ipykernel install --user --name=venv_name` to register your environment

## Support

If you encounter issues:

1. Check this README and troubleshooting section first
2. Look for similar issues in the repository
3. Create a new issue with:
   - Your operating system
   - Python version
   - Error message (if any)
   - Steps to reproduce the problem

If trouble persists, contact: **saiguruinukurthi@gmail.com**

## Additional Resources

### General Python & ML
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Jupyter Documentation](https://jupyter.readthedocs.io/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Deep Learning & CNNs (Module 6)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CNN Explainer Interactive Tool](https://poloclub.github.io/cnn-explainer/)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

---

**Happy Learning!**
