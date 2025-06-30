#!/bin/bash

# MLDevops Repository Push Script
# This script helps you push your MLDevops project to a Git repository

set -e

echo "🚀 MLDevops Repository Push Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Check if git is initialized
if [ ! -d ".git" ]; then
    print_error "Git repository not initialized. Run 'git init' first."
    exit 1
fi

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    print_warning "You have uncommitted changes. Please commit them first:"
    git status --short
    echo ""
    read -p "Do you want to commit these changes? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "Update: $(date)"
        print_status "Changes committed successfully."
    else
        print_error "Please commit your changes before pushing."
        exit 1
    fi
fi

# Get repository URL
print_header "Repository Configuration"
echo "Choose your Git hosting service:"
echo "1) GitHub"
echo "2) GitLab"
echo "3) Bitbucket"
echo "4) Custom URL"
echo "5) Skip remote setup (just show commands)"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        read -p "Enter your GitHub username: " username
        read -p "Enter repository name: " repo_name
        remote_url="https://github.com/$username/$repo_name.git"
        ;;
    2)
        read -p "Enter your GitLab username: " username
        read -p "Enter repository name: " repo_name
        remote_url="https://gitlab.com/$username/$repo_name.git"
        ;;
    3)
        read -p "Enter your Bitbucket username: " username
        read -p "Enter repository name: " repo_name
        remote_url="https://bitbucket.org/$username/$repo_name.git"
        ;;
    4)
        read -p "Enter custom repository URL: " remote_url
        ;;
    5)
        print_header "Manual Push Commands"
        echo "Run these commands manually:"
        echo ""
        echo "# Add remote repository"
        echo "git remote add origin <your-repo-url>"
        echo ""
        echo "# Push to remote repository"
        echo "git branch -M main"
        echo "git push -u origin main"
        echo ""
        echo "# For subsequent pushes"
        echo "git push"
        exit 0
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Add remote repository
print_status "Adding remote repository..."
git remote add origin "$remote_url" 2>/dev/null || git remote set-url origin "$remote_url"

# Set main as default branch
print_status "Setting main as default branch..."
git branch -M main

# Push to remote repository
print_status "Pushing to remote repository..."
if git push -u origin main; then
    print_status "✅ Successfully pushed to remote repository!"
    echo ""
    print_header "Repository Information"
    echo "Remote URL: $remote_url"
    echo "Branch: main"
    echo ""
    print_header "Next Steps"
    echo "1. Visit your repository URL to verify the push"
    echo "2. Set up CI/CD integration if needed"
    echo "3. Share the repository with your team"
    echo ""
    print_header "Repository Contents"
    echo "✅ Complete MLDevops pipeline"
    echo "✅ XGBoost model training and serving"
    echo "✅ Kubeflow automation"
    echo "✅ Jenkins CI/CD pipeline"
    echo "✅ AWS EKS infrastructure"
    echo "✅ Comprehensive documentation"
    echo ""
    print_status "🎉 Your MLDevops project is now ready for collaboration!"
else
    print_error "Failed to push to remote repository."
    echo ""
    print_header "Troubleshooting"
    echo "1. Check your internet connection"
    echo "2. Verify your Git credentials"
    echo "3. Ensure the repository exists and you have access"
    echo "4. Try running: git push -u origin main --force (if needed)"
    exit 1
fi

echo ""
print_header "Additional Commands"
echo "# Check remote repository"
echo "git remote -v"
echo ""
echo "# Check repository status"
echo "git status"
echo ""
echo "# View commit history"
echo "git log --oneline"
echo ""
echo "# Pull latest changes"
echo "git pull origin main" 