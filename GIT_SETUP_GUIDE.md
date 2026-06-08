# 🚀 GIT SETUP GUIDE — Step-by-Step Push Instructions

Follow this guide to go from files on your laptop to a live GitHub repo.

---

## Prerequisites

```bash
# Check if git is installed
git --version

# Configure identity (one-time setup)
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

---

## Step 1: Create the Local Repo

```bash
# Navigate to the folder where you want the project
cd ~/Documents

# Create project folder and enter it
mkdir ml-journey && cd ml-journey

# Initialize git
git init
```

---

## Step 2: Copy All Files

Place all the files from this README set into your `ml-journey/` folder.  
The structure should look like:

```
ml-journey/
├── README.md
├── .gitignore
├── 01-intro-to-ml/
│   ├── README.md
│   └── linear_regression_scratch.py
├── 02-ml-models/
│   └── README.md
├── 03-implementations/
│   └── README.md
├── 04-model-saving-unsupervised/
│   └── README.md
├── 05-neural-networks/
│   └── README.md
├── 06-deployment/
│   ├── README.md
│   ├── Dockerfile
│   └── docker-compose.yml
└── resources/
    └── README.md
```

---

## Step 3: Create .gitignore

Create a file called `.gitignore` in the root:

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
venv/
.env
*.egg-info/

# Jupyter
.ipynb_checkpoints/
*.ipynb        # optional: exclude notebooks if you prefer .py files

# ML Models (never push large model files)
*.h5
*.hdf5
*.pkl
*.pickle
*.pt
*.pth
*.joblib
*.onnx
*.pb
models/
saved_models/

# Datasets (too large for git)
*.csv
*.json
data/
datasets/
raw/

# Plots output
plots/*.png
plots/*.jpg

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
EOF
```

---

## Step 4: First Commit

```bash
# Stage all files
git add .

# Verify what's being staged (should NOT include models or data)
git status

# Commit
git commit -m "Initial commit: ML Journey documentation and code structure"
```

---

## Step 5: Create GitHub Repo

1. Go to **https://github.com/new**
2. Repository name: `ml-journey`
3. Description: `Structured ML learning journey — from scratch to deployment`
4. Choose Public or Private
5. ⚠️ **Do NOT check** "Add a README file" (we have our own)
6. ⚠️ **Do NOT check** "Add .gitignore" (we have our own)
7. Click **Create repository**

---

## Step 6: Connect and Push

GitHub will show you a page with commands. Use these:

```bash
# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/ml-journey.git

# Rename branch to main (modern convention)
git branch -M main

# Push to GitHub
git push -u origin main
```

The `-u` flag sets `origin main` as the default upstream — after this, you can just type `git push`.

---

## Step 7: Ongoing Workflow (After Each Study Session)

```bash
# 1. After adding/modifying files
git status                        # See what changed
git diff                          # See exact changes

# 2. Stage changes
git add 03-implementations/       # Stage a whole folder
git add 03-implementations/svm_iris.py   # Or a single file
git add .                         # Stage everything (be careful)

# 3. Commit with a meaningful message
git commit -m "Add SVM implementation with decision boundary plots"

# 4. Push
git push
```

---

## Recommended Commit Message Format

```
<type>: <short description>

Types:
  feat     → new feature / file
  docs     → documentation update
  fix      → bug fix
  refactor → code cleanup
  add      → adding content/notes

Examples:
  feat: Add CNN training script for MNIST
  docs: Complete Topic 4 notes on kernels
  fix: Correct OLS formula in scratch implementation
  add: Decision boundary plots for all 3 Iris models
```

---

## Step 8: Branching (for New Topics)

When your senior gives you a new topic:

```bash
# Create a new branch
git checkout -b topic/07-transformers

# Work on it, then commit
git add .
git commit -m "feat: Add Transformer architecture notes"

# Push the branch
git push origin topic/07-transformers

# When complete, merge into main via GitHub Pull Request
# OR merge locally:
git checkout main
git merge topic/07-transformers
git push
```

---

## Verification Checklist Before Each Push

- [ ] No `.h5`, `.pkl`, `.pt` model files staged (`git status`)
- [ ] No large CSV/data files staged
- [ ] `.gitignore` is in root and working
- [ ] Commit message is descriptive
- [ ] Code runs without errors

---

## Useful Git Commands

```bash
git log --oneline          # See commit history
git diff HEAD~1 HEAD       # See what changed in last commit
git checkout -- file.py    # Discard uncommitted changes to a file
git stash                  # Temporarily save uncommitted changes
git stash pop              # Restore stashed changes
git remote -v              # Show remote URLs
```
