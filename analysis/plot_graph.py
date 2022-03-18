import matplotlib.pyplot as plt
VERBOSE = False
from matplotlib import cm
import networkx as nx
from analysis.graph_utils import get_path_from_source, get_zero_graph, get_target_of_type

def plot_dms_graph(G):
    dd = {}
    for node in G.nodes:
        if node == 0:
            dd[node] = 0
        elif G.nodes[node]['output'] != 0:
            dd[node] = 3
        elif len(G[node]) == 4:
            dd[node] = 1
        else:
            dd[node] = 2

    nx.set_node_attributes(G, dd, 'subset')
    pos = nx.multipartite_layout(G)
    edges, edges_color = zip(*nx.get_edge_attributes(G, 'edge_type').items())
    edge_markers = ['dashed' if e < 0 else 'solid' for e in edges_color]
    # edges_color = np.abs(edges_color)
    nodes, nodes_color = zip(*nx.get_node_attributes(G, 'output').items())
    nonzero_nodes = [nodes[i] for i in range(len(nodes)) if nodes_color[i] != 0]
    nonzero_colors = [nodes_color[i] for i in range(len(nodes)) if nodes_color[i] != 0]
    zero_edges = [edges[i] for i in range(len(edges)) if edges_color[i] == 0]
    positive_edges = [edges[i] for i in range(len(edges)) if edges_color[i] > 0]
    positive_colors = [edges_color[i] for i in range(len(edges)) if edges_color[i] > 0]
    negative_edges = [edges[i] for i in range(len(edges)) if edges_color[i] < 0]
    negative_colors = [abs(edges_color[i]) for i in range(len(edges)) if edges_color[i] < 0]


    plt.clf()
    node_size = 500
    min_target_margin = 5
    nx.draw(G, pos, node_color='white', node_size=node_size, edgelist=[], edgecolors='black');
    nx.draw_networkx_nodes(G, pos, nonzero_nodes, node_color=nonzero_colors, cmap=cm.Reds, vmin=0, vmax=3, node_size=node_size, edgecolors='black');
    nx.draw_networkx_edges(G, pos, connectionstyle='arc3, rad = 0.2', edgelist=zero_edges, edge_color='black', width=2, arrowsize=20,  min_source_margin=10, min_target_margin=min_target_margin);
    nx.draw_networkx_edges(G, pos, connectionstyle='arc3, rad = 0.2', edgelist=positive_edges, edge_color=positive_colors, width=2, arrowsize=20, edge_cmap=cm.Reds, edge_vmin=0, edge_vmax=max(edges_color) + 1,  min_source_margin=10, min_target_margin=min_target_margin);
    nx.draw_networkx_edges(G, pos, connectionstyle='arc3, rad = 0.2', edgelist=negative_edges, edge_color=negative_colors, width=2, arrowsize=20, edge_cmap=cm.Reds, edge_vmin=0, edge_vmax=max(edges_color) + 1, style='dashed',  min_source_margin=10, min_target_margin=min_target_margin);
    nx.draw_networkx_nodes(G, pos, [0], node_color='red', node_size=node_size);



def plot_interval_discrimination_graph(G, bipartite=True):
    G = G.copy()
    plt.clf();
    if len(G) == 2:
        pos = nx.spring_layout(G, seed=0)
    else:
        pos = nx.circular_layout(G)
    if bipartite:
        if len(G) == 2:
            pos = nx.bipartite_layout(G, G.nodes())
        else:
            zero_path = get_path_from_source(get_zero_graph(G), 0)
            dd = {}
            for node in G.nodes:
                if node in zero_path:
                    dd[node] = 0
                elif G.nodes[node]['output'] != 0:
                    dd[node] = 2
                else:
                    dd[node] = 1

            nx.set_node_attributes(G, dd, 'subset')
            pos = nx.multipartite_layout(G)
            sorted_positions = sorted([pos[node] for node in zero_path], key= lambda x: x[1])
            for i, node in enumerate(zero_path):
                pos[node] = sorted_positions[i]

    edges, edges_color = zip(*nx.get_edge_attributes(G, 'edge_type').items())
    nodes, nodes_color = zip(*nx.get_node_attributes(G, 'output').items())

    set_edges = [edges[i] for i in range(len(edges)) if edges_color[i] == 2]
    ready_edges = [edges[i] for i in range(len(edges)) if edges_color[i] == 4]
    zero_edges = [edges[i] for i in range(len(edges)) if edges_color[i] == 0]

    set_edges, ready_edges = ready_edges, set_edges
    node_size = 1000
    nx.draw(G, pos, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=set_edges, edge_color='gold', width=2, arrowsize=20,  min_target_margin=15);
    nx.draw(G, pos, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=ready_edges, edge_color='orange', width=2, arrowsize=20,  min_target_margin=15);
    nx.draw(G, pos, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=zero_edges, edge_color='black', width=2, arrowsize=20,  min_target_margin=15);
    labels = nx.get_node_attributes(G, 'skip')
    nx.draw_networkx_nodes(G, pos, nodes, node_color=nodes_color, cmap='PiYG', vmin=-1, vmax=1, edgecolors='black', node_size=node_size);
    nx.draw_networkx_nodes(G, pos, [0], node_color='red', node_size=node_size);
    nx.draw_networkx_labels(G, pos, labels)


def plot_interval_reproduction_graph(G, bipartite=True):
    G = G.copy()
    plt.clf();
    if len(G) == 2:
        pos = nx.spring_layout(G, seed=0)
    else:
        pos = nx.circular_layout(G)
    if bipartite:
        if len(G) == 2:
            pos = nx.bipartite_layout(G, G.nodes())
        else:
            zero_path = get_path_from_source(get_zero_graph(G), 0)
            dd = {}
            for node in G.nodes:
                if node in zero_path:
                    dd[node] = 0
                elif G.nodes[node]['output'] != 0:
                    dd[node] = 2
                else:
                    dd[node] = 1

            nx.set_node_attributes(G, dd, 'subset')
            pos = nx.multipartite_layout(G)
            sorted_positions = sorted([pos[node] for node in zero_path], key= lambda x: x[1])
            for i, node in enumerate(zero_path):
                pos[node] = sorted_positions[i]

            rest_order = [get_target_of_type(G, node, 2) for node in zero_path]
            rest_order = [node for node in rest_order if node is not None]
            not_zero_path = [node for node in G.nodes if G.nodes[node]['subset'] == 1 and node not in rest_order] + rest_order
            # not_zero_path = sorted([node for node in G.nodes if node not in zero_path])
            sorted_positions = sorted([pos[node] for node in not_zero_path], key= lambda x: x[1])
            # for i, node in enumerate(not_zero_path):
            #     pos[node] = sorted_positions[i]

    edges, edges_color = zip(*nx.get_edge_attributes(G, 'edge_type').items())
    nodes, nodes_color = zip(*nx.get_node_attributes(G, 'output').items())
    labels = nx.get_node_attributes(G, 'skip')
    nx.draw(G, pos, labels=labels, node_color='grey', connectionstyle='arc3, rad = 0.2', edgelist=edges, edge_color=edges_color, edge_cmap=cm.hot, edge_vmin=0, edge_vmax=max(edges_color) + 1, width=2, arrowsize=20,  min_source_margin=10, min_target_margin=15);
    nx.draw_networkx_nodes(G, pos, nodes, node_color=nodes_color, cmap='PiYG', vmin=-1, vmax=1, edgecolors='black', node_size=1200);
    nx.draw_networkx_nodes(G, pos, [0], node_color='red', node_size=1200);

