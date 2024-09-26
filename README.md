# Import libraries
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

# Load the data
df = pd.read_csv(r"C:\Users\SHUBHAM GUPTA\Desktop\datasets\cosmetics.csv")

# Check the first five rows 
display(df.sample(5))

# Inspect the types of products
product_counts=df['Label'].value_counts()
display(product_counts)

# Focus on one product category and one skin type
# Filter for moisturizers
moisturizers = df[df['Label'] == 'Moisturizer']

# Filter for dry skin as well
moisturizers_dry = moisturizers[moisturizers['Dry']==1]

# Reset index
moisturizers_dry = moisturizers_dry.reset_index(drop=True)
display(moisturizers_dry.head())

# Tokenizing the Ingredient
# Initialize dictionary, list, and initial index
ingredient_idx = {}
corpus = []
idx = 0

# For loop for tokenization
for i in range(len(moisturizers_dry)):    
    ingredients = moisturizers_dry['Ingredients'][i]
    # Convert ingredients to lowercase and split by ', '
    tokens = ingredients.lower().split(', ')
    # Append tokens (list of ingredients) to the corpus
    corpus.append(tokens) 
    # Add ingredients to the dictionary if not already present
    for ingredient in tokens:
        if ingredient not in ingredient_idx:
            ingredient_idx[ingredient] = idx
            idx += 1
            
# Check the result 
print("The index for decyl oleate is", ingredient_idx.get('decyl oleate', 'Not found'))

# Initializing a document-term matrix(DTM)
# Get the number of items and tokens 
M = len(moisturizers_dry)
N = len(ingredient_idx)

# Initialize a matrix of zeros
A = np.zeros((M,N))

# Creating a counter Function
# Define the oh_encoder function
def oh_encoder(tokens, ingredient_idx):
    x = np.zeros(N)    
     # Get the index for each ingredient
    for token in tokens:
        if token in ingredient_idx:
            x[ingredient_idx[token]]= 1
        # Put 1 at the corresponding indices
    return x

# Cosmetic-Ingredient Matrix
# Make a document-term matrix

for i, tokens in enumerate(corpus):
    A[i, :] = oh_encoder(tokens, ingredient_idx)
print(A)


# Dimension reduction with t-SNE
model = TSNE (n_components=2, learning_rate=200, random_state=42)
tsne_features = model.fit_transform(A)

# Make X, Y columns 
moisturizers_dry['X'] = tsne_features[:,0]
moisturizers_dry['Y'] = tsne_features[:,1]

# Let's map the items with bokeh
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource 

# Make a source and a scatter plot  
source = ColumnDataSource(moisturizers_dry)
plot = figure(x_axis_label = 'TSNE 1', 
              y_axis_label = 'TSNE 2', 
              width = 500, height = 400)
plot.circle(x = 'X', 
    y = 'Y', 
    source = source, 
    size = 10, color = '#FF7373', alpha = .8)

# Adding the hover tool
# Create a HoverTool object
from bokeh.models import HoverTool
hover = HoverTool(tooltips = [("Item",'@Name'),
                              ('Brand','@Brand'),
                              ('Price','$@Price'),
                              ("Rank",'@Rank')])
plot.add_tools(hover)

# Mapping the cosmetic items
# Plot the map
show(plot)

# Comparing two products
# Print the ingredients of two similar cosmetics
cosmetic_1 = moisturizers_dry[moisturizers_dry['Name'] == "Color Control Cushion Compact Broad Spectrum SPF 50+"]
cosmetic_2 = moisturizers_dry[moisturizers_dry['Name'] == "BB Cushion Hydra Radiance SPF 50"]

# Display each item's data and ingredients
display(cosmetic_1)
print(cosmetic_1.Ingredients.values)
display(cosmetic_2)
print(cosmetic_2.Ingredients.values)
