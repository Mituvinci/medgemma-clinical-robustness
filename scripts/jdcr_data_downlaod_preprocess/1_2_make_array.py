file_path = "urls.txt"

with open(file_path, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

print("VOLUME_URLS = [")
for url in urls:
    print(f'    "{url}",')
print("]")
