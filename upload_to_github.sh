#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "   DEEP LEARNING & COMPUTER VISION SUITE - GITHUB DEPLOYMENT"
echo "   TensorFlow/Keras | OpenCV | Scikit-learn | Pandas | Streamlit"
echo "======================================================================"

python3 copy_assets.py 2>/dev/null || python copy_assets.py 2>/dev/null || true

if [ ! -d ".git" ]; then
    echo "[INFO] Initializing new Git repository..."
    git init
fi

git add .
git commit -m "feat: Deep Learning & Computer Vision Suite (TensorFlow, OpenCV, Scikit-Learn, Pandas) [>92% Text Acc, 30+ FPS CV]" || true
git branch -M main

read -p "Enter GitHub Repository URL [default: https://github.com/MidhuBabyBrankottil/deep-learning-cv-suite.git]: " REPO_URL
REPO_URL=${REPO_URL:-"https://github.com/MidhuBabyBrankottil/deep-learning-cv-suite.git"}

git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

echo ""
echo "[INFO] Pushing main branch to: $REPO_URL"
git push -u origin main

echo ""
echo "[SUCCESS] Project uploaded to GitHub!"
