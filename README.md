Project Overview
This project involves analyzing a dataset of cosmetic products, focusing on moisturizers for dry skin, to understand the relationship between their ingredients using t-SNE (t-Distributed Stochastic Neighbor Embedding). The goal is to visualize and compare cosmetic products based on their ingredients, providing insights into similar products.

Technologies Used
Python Libraries:
pandas – for data manipulation and analysis
numpy – for numerical computations
scikit-learn – for t-SNE dimensionality reduction
bokeh – for interactive visualization
Dataset Information
The dataset, cosmetics.csv, contains the following columns:

Name: Name of the product
Brand: Brand of the product
Label: Type of product (e.g., Moisturizer)
Ingredients: List of ingredients used in the product
Price: Price of the product
Rank: Product rating or rank
Dry: Indicates if the product is suitable for dry skin (1 = Yes, 0 = No)
Key Steps and Code Workflow
1. Loading and Inspecting the Data
python
Copy code
# Load the data
df = pd.read_csv("path/to/cosmetics.csv")

# Display the first 5 rows of the dataset
display(df.sample(5))
2. Inspecting Product Types
python
Copy code
# Inspect the types of products in the dataset
product_counts = df['Label'].value_counts()
display(product_counts)
3. Filtering Moisturizers for Dry Skin
python
Copy code
# Filter for moisturizers and dry skin products
moisturizers = df[df['Label'] == 'Moisturizer']
moisturizers_dry = moisturizers[moisturizers['Dry'] == 1]
moisturizers_dry = moisturizers_dry.reset_index(drop=True)
display(moisturizers_dry.head())
4. Tokenizing the Ingredients
We tokenize the ingredients of each product to prepare the data for further analysis.

python
Copy code
# Tokenizing ingredients and building a mapping dictionary
ingredient_idx = {}
corpus = []
idx = 0

for i in range(len(moisturizers_dry)):    
    ingredients = moisturizers_dry['Ingredients'][i]
    tokens = ingredients.lower().split(', ')
    corpus.append(tokens)
    for ingredient in tokens:
        if ingredient not in ingredient_idx:
            ingredient_idx[ingredient] = idx
            idx += 1
5. Creating the Document-Term Matrix (DTM)
A document-term matrix (DTM) is created to represent the frequency of ingredients in each product.

python
Copy code
M = len(moisturizers_dry)
N = len(ingredient_idx)
A = np.zeros((M, N))

# Function to encode ingredients
def oh_encoder(tokens, ingredient_idx):
    x = np.zeros(N)
    for token in tokens:
        if token in ingredient_idx:
            x[ingredient_idx[token]] = 1
    return x

# Fill the matrix with encoded values
for i, tokens in enumerate(corpus):
    A[i, :] = oh_encoder(tokens, ingredient_idx)
6. Dimensionality Reduction with t-SNE
t-SNE is used to reduce the dimensions of the DTM for visualization.

python
Copy code
from sklearn.manifold import TSNE
model = TSNE(n_components=2, learning_rate=200, random_state=42)
tsne_features = model.fit_transform(A)

# Add t-SNE features to the dataframe
moisturizers_dry['X'] = tsne_features[:, 0]
moisturizers_dry['Y'] = tsne_features[:, 1]
7. Visualizing with Bokeh
We use Bokeh to create an interactive scatter plot, mapping the products in 2D space based on their t-SNE features.

python
Copy code
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool

# Create the plot
source = ColumnDataSource(moisturizers_dry)
plot = figure(x_axis_label='TSNE 1', y_axis_label='TSNE 2', width=500, height=400)
plot.circle(x='X', y='Y', source=source, size=10, color='#FF7373', alpha=0.8)

# Adding hover tool for detailed information
hover = HoverTool(tooltips=[("Item", '@Name'), ('Brand', '@Brand'), ('Price', '$@Price'), ("Rank", '@Rank')])
plot.add_tools(hover)

# Show the plot
show(plot)
8. Comparing Two Products
Finally, we compare the ingredients of two similar products by extracting their details.

python
Copy code
# Comparing two similar products
cosmetic_1 = moisturizers_dry[moisturizers_dry['Name'] == "Product 1"]
cosmetic_2 = moisturizers_dry[moisturizers_dry['Name'] == "Product 2"]

# Display ingredients for comparison
display(cosmetic_1)
print(cosmetic_1.Ingredients.values)
display(cosmetic_2)
print(cosmetic_2.Ingredients.values)
Visualizations and Insights
After applying t-SNE for dimensionality reduction, we created a 2D scatter plot to visualize the similarities and differences between cosmetic products based on their ingredients. Products that are similar in ingredients appear closer together in the plot.

Conclusion
This project demonstrates how dimensionality reduction techniques like t-SNE can be used to analyze and visualize high-dimensional data, such as cosmetic product ingredients. The ability to compare products based on their ingredients can help in identifying trends, similarities, and potential gaps in the market.

Future Improvements
Implement a recommendation system based on ingredients similarity.
Extend the analysis to other product categories.
Improve the visualization by adding more interactive elements.
Let me know if you need further modifications or explanations!
