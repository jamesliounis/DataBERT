# Remove old build/dist files
rm -r dist build

# Build new distribution files with the updated project name
python setup.py sdist bdist_wheel

# Attempt to upload the package again to PyPI
twine upload dist/*

