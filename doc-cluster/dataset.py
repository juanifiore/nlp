import kagglehub

# Download latest version
path = kagglehub.dataset_download("dylanjcastillo/news-headlines-2024")

print("Path to dataset files:", path)