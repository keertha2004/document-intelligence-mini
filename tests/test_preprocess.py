from preprocess import clean_text, extract_metadata

sample = "Apple launches new iPhones in 2025 with price $999."

print("Original:", sample)
print("Cleaned :", clean_text(sample))
print("Metadata:", extract_metadata(sample))
