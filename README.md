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
├── Mod2_Project.ipynb                          # Module 2 project
├── Mod3_Lab1_Understanding_Distance_metrics_and_Introduction_to_KNN.ipynb  # Distance metrics & KNN
├── Mod3_Lab2_Using_KNN_for_Text_Classification.ipynb      # KNN for text analysis
├── Mod3_Lab3_Implementing_KNN_from_scratch_and_visualize_Algorithm_performance.ipynb  # KNN from scratch
├── Mod4_Lab1_Perceptron_and_Gradient_Descent.ipynb        # Perceptron & gradient descent
├── Mod4_Lab2_Introduction_to_Gradient_Descent.ipynb       # Advanced gradient descent
├── Mod4_Lab3_Gradient_Descent.ipynb            # Advanced gradient descent techniques
├── Mod5_Lab1_Linear_Regression_MSE_and_Polynomial_Regression.ipynb  # Linear regression & polynomial regression
├── Mod5_Lab2_Loss_Functions.ipynb              # Loss functions analysis
├── Mod5_Lab3_Clustering.ipynb                  # Clustering algorithms (K-Means, Hierarchical, DBSCAN)
├── Mod6_Lab1_Implementing_forward_propagation_and_back_propagation.ipynb  # Neural networks fundamentals
├── car_evaluation.csv                          # Car evaluation dataset
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
# For Module 1 notebooks
python -m venv .venv_mod1
.venv_mod1\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn

# For Module 2 notebooks
python -m venv .venv_mod2
.venv_mod2\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn plotly

# For Module 3 notebooks
python -m venv .venv_mod3
.venv_mod3\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn scipy nltk

# For Module 4 notebooks
python -m venv .venv_mod4
.venv_mod4\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# For Module 5 notebooks
python -m venv .venv_mod5
.venv_mod5\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# For Module 6 notebooks
python -m venv .venv_mod6
.venv_mod6\Scripts\activate  # Windows
pip install numpy matplotlib scikit-learn tensorflow keras torch torchvision
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
tensorflow>=2.20.0  # For deep learning (Module 6)
keras>=3.11.0       # For neural networks
torch>=2.8.0        # PyTorch for deep learning
torchvision>=0.23.0 # Computer vision with PyTorch
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

## Module-Specific Notes

### Module 1: Foundations
- Focus on basic ML concepts and data preprocessing
- Requires: pandas, numpy, matplotlib, seaborn

### Module 2: Dimensionality Reduction
- Covers PCA and manifold learning
- Requires: scikit-learn, plotly for interactive plots

### Module 3: K-Nearest Neighbors
- Implementation from scratch and performance analysis
- Requires: scipy for distance metrics, nltk for text processing

### Module 4: Gradient Descent
- Advanced optimization techniques
- Requires: scipy for optimization functions

### Module 5: Linear Regression & Clustering
- Linear regression, polynomial regression, and loss functions
- Clustering algorithms: K-Means, Hierarchical, DBSCAN
- Requires: scikit-learn, matplotlib, seaborn for visualization

### Module 6: Neural Networks
- Forward propagation and back propagation implementation
- Neural network fundamentals
- Requires: tensorflow, keras, torch for deep learning

## Learning Path

1. **Start with**: `Mod0_Linear_Algebra.ipynb` and `Mod0_Probability_Basics.ipynb`
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

## Support

If you encounter issues:

1. Check this README first
2. Look for similar issues in the repository
3. Create a new issue with:
   - Your operating system
   - Python version
   - Error message (if any)
   - Steps to reproduce the problem

If trouble persists, contact: **saiguruinukurthi@gmail.com**

## Additional Resources

- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Jupyter Documentation](https://jupyter.readthedocs.io/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

**Happy Learning!**
