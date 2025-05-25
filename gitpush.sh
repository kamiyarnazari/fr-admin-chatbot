#!/bin/bash

echo "📦 Staging all changes..."
git add .

echo "📝 Commit message:"
read message

git commit -m "$message"

echo "🚀 Pushing to main..."
git push origin main

echo "✅ Done!"
