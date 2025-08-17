import pandas as pd

df = pd.read_json("data/News_Category_Dataset_v2.json", lines=True)
df = df[['headline', 'category']]
df.rename(columns={'headline': 'text', 'category': 'label'}, inplace=True)

# Keep only top 5 categories
top_cats = df['label'].value_counts().nlargest(5).index
df = df[df['label'].isin(top_cats)]

df.to_csv("data/news.csv", index=False)
print("✅ news.csv saved with shape:", df.shape)
print("Categories:", df['label'].unique())
