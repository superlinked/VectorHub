#!/bin/bash

# Loop through all tracked files
git ls-tree -r --name-only HEAD | while read filename; do
  # Get the last commit date for this file
  last_commit_date=$(git log -1 --format="%ai" -- "$filename")
  # Use 'touch' to update the file's timestamp
  touch -d "$last_commit_date" "$filename"
done