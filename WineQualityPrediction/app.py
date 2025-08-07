import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, Circle, ArrowStyle

plt.figure(figsize=(20, 15))
plt.title("Data Science MentorBot: Multi-Agent RAG System Architecture\n", fontsize=16, fontweight='bold')

# Create a directed graph
G = nx.DiGraph()

# Define nodes with types and positions
nodes = {
    "User": {"type": "input", "pos": (1, 5), "color": "#FF6B6B"},
    "Coordinator\nAgent": {"type": "agent", "pos": (4, 5), "color": "#4ECDC4"},
    "Q&A\nAgent": {"type": "agent", "pos": (7, 7), "color": "#45B7D1"},
    "Practice Problem\nAgent": {"type": "agent", "pos": (7, 5), "color": "#45B7D1"},
    "Socratic Dialogue\nAgent": {"type": "agent", "pos": (7, 3), "color": "#45B7D1"},
    "Dataset Analysis\nAgent": {"type": "agent", "pos": (10, 5), "color": "#45B7D1"},
    "Feedback\nAgent": {"type": "agent", "pos": (4, 3), "color": "#45B7D1"},
    "Knowledge Bases": {"type": "database", "pos": (4, 7), "color": "#FFC154"},
    "User Data\n(SQLite)": {"type": "database", "pos": (1, 3), "color": "#A2D729"},
    "Tools": {"type": "tools", "pos": (10, 3), "color": "#C792EA"},
    "Web App": {"type": "interface", "pos": (1, 7), "color": "#82AAFF"}
}

# Add nodes to graph
for node, attrs in nodes.items():
    G.add_node(node, pos=attrs["pos"], color=attrs["color"])

# Define edges with labels
edges = [
    ("User", "Coordinator\nAgent", "Text/CSV Upload"),
    ("Coordinator\nAgent", "Q&A\nAgent", "Theory Questions"),
    ("Coordinator\nAgent", "Practice Problem\nAgent", "Problem Requests"),
    ("Coordinator\nAgent", "Socratic Dialogue\nAgent", "Critical Thinking Qs"),
    ("Coordinator\nAgent", "Dataset Analysis\nAgent", "Dataset Processing"),
    ("Q&A\nAgent", "Knowledge Bases", "RAG Retrieval"),
    ("Practice Problem\nAgent", "Knowledge Bases", "RAG Retrieval"),
    ("Socratic Dialogue\nAgent", "Knowledge Bases", "RAG Retrieval"),
    ("Dataset Analysis\nAgent", "Knowledge Bases", "RAG Retrieval"),
    ("Dataset Analysis\nAgent", "Tools", "Uses pandas/scikit-learn"),
    ("Feedback\nAgent", "Knowledge Bases", "Updates KB"),
    ("Feedback\nAgent", "User Data\n(SQLite)", "Stores Feedback"),
    ("Web App", "User", "UI Interaction"),
    ("User", "Web App", "Input/Upload"),
    ("Web App", "Coordinator\nAgent", "Routes Requests"),
    ("Coordinator\nAgent", "Feedback\nAgent", "User Feedback"),
    ("Q&A\nAgent", "User", "Explanations"),
    ("Practice Problem\nAgent", "User", "Quizzes/Challenges"),
    ("Socratic Dialogue\nAgent", "User", "Guiding Questions"),
    ("Dataset Analysis\nAgent", "User", "Insights/Code")
]

for edge in edges:
    G.add_edge(edge[0], edge[1], label=edge[2])

# Draw nodes with different shapes
pos = nx.get_node_attributes(G, 'pos')
colors = nx.get_node_attributes(G, 'color')

# Draw different node types with shapes
for node in G.nodes():
    if nodes[node]["type"] == "input":
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='s', node_size=5000, node_color=colors[node])
    elif nodes[node]["type"] == "agent":
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='d', node_size=5000, node_color=colors[node])
    elif nodes[node]["type"] == "database":
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='s', node_size=5000, node_color=colors[node])
    elif nodes[node]["type"] == "tools":
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='^', node_size=5000, node_color=colors[node])
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='o', node_size=5000, node_color=colors[node])

# Draw edges with labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, edge_color='gray', arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Add legend
legend_elements = [
    Rectangle((0,0), 1, 1, color='#FF6B6B', label='User Input/Output'),
    Rectangle((0,0), 1, 1, color='#45B7D1', label='Specialized Agents'),
    Rectangle((0,0), 1, 1, color='#4ECDC4', label='Coordinator Agent'),
    Rectangle((0,0), 1, 1, color='#FFC154', label='Knowledge Bases (FAISS)'),
    Rectangle((0,0), 1, 1, color='#A2D729', label='User Data Storage'),
    Rectangle((0,0), 1, 1, color='#C792EA', label='Tools (pandas, scikit-learn)'),
    Rectangle((0,0), 1, 1, color='#82AAFF', label='Web Interface')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add component details
component_details = {
    "Knowledge Bases": "- FAISS indexes with Sentence-BERT\n- Textbooks, Kaggle, scikit-learn docs\n- arXiv papers, Stack Overflow",
    "Dataset Analysis\nAgent": "- Pandas for preprocessing\n- Matplotlib/seaborn visuals\n- Algorithm recommendations\n- Feature engineering",
    "Tools": "- pandas (describe, corr)\n- scikit-learn (StandardScaler)\n- imblearn (SMOTE)\n- Docker for execution",
    "Multi-Agent\nCoordination": "- LangGraph for workflows\n- Dynamic agent activation\n- Context sharing\n- Feedback integration"
}

for node, detail in component_details.items():
    x, y = pos[node]
    plt.text(x + 1.2, y - 0.3, detail, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

# Add architecture notes
plt.text(1, 1, "Key Architectural Features:\n"
               "- Multi-agent RAG system with specialized agents\n"
               "- CPU-efficient design (quantized DistilBERT/TinyLLaMA)\n"
               "- Chunked processing for 2GB datasets\n"
               "- Feedback-driven knowledge base updates\n"
               "- Collaborative agent reasoning via LangGraph",
         fontsize=10, bbox=dict(facecolor='#f0f0f0', alpha=0.8))

plt.grid(False)
plt.axis('off')
plt.tight_layout()
plt.savefig("mentorbot_architecture.png", dpi=300, bbox_inches='tight')
plt.show()