import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, from_networkx
from bokeh.models import HoverTool
from bokeh.io import show, output_file
import pymongo
# MongoDB Connection


client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection

titles = []
embeddings = []

# Collect titles and embeddings
for doc in collection.find():
    title = doc.get('title')
    embedding = doc.get('abstract_embedding')
    if title is not None and embedding is not None:
        titles.append(title)
        embeddings.append(embedding)

threshold = 0.7

# Build graph
G = nx.Graph()

# Add nodes (titles)
for i, title in enumerate(titles):
    G.add_node(i, title=title)

# Add edges based on similarity
num_titles = len(titles)
for i in range(num_titles):
    for j in range(i + 1, num_titles):
        similarity = cosine_similarity(np.array(embeddings[i]).reshape(1, -1), np.array(embeddings[j]).reshape(1, -1))[0][0]
        if similarity > threshold:
            G.add_edge(i, j, weight=similarity)

# Create Bokeh plot
plot = figure(title="Thesis Knowledge Graph", x_range=(-1.5, 1.5), y_range=(-1.5, 1.5),
              tools="pan,wheel_zoom,box_zoom,reset,hover,save")



# Add hover tool
hover = HoverTool()
hover.tooltips = [("Title", "@title")]
plot.add_tools(hover)

# Convert NetworkX graph to Bokeh graph
graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.data_source.data['size'] = [20] * len(G.nodes)  # Set default node size
graph_renderer.node_renderer.glyph.size = 20  # Change node size

plot.renderers.append(graph_renderer)

output_file("thesis_knowledge_graph.html")


show(plot)
