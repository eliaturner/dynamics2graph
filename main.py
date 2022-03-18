import matplotlib.pyplot as plt
from analysis.dynamics_to_full_graph import create_lace_from_dynamics, connect_lace_to_graph, merge_components
from analysis.compress_graph import remove_uninformative_paths, merge_edges, merge_zero_one_edges
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from analysis.graph_utils import mark_output_nodes, get_path_from_source, get_target_of_type, mark_output_nodes_rsg
from analysis.plot_graph import plot_interval_discrimination_graph, plot_interval_reproduction_graph, plot_dms_graph
from analysis.dynamics_to_full_graph import DISTANCE_THRESHOLD

import pickle as pkl


def load_pickle(file_name):
    print('loading', file_name)
    with open(file_name, 'rb') as handle:
        file = pkl.load(handle)
    return file


def dump_pickle(file_name, var):
    with open(file_name, 'wb') as handle:
        pkl.dump(var, handle)


def step2(G, rule_set):
    nx.set_node_attributes(G, 1, name='skip')
    G_size = len(G) + 1
    while len(G) < G_size:
        G_size = len(G)
        for rule in rule_set:
            G = rule(G)

    return G

def scale_rsg_data(trials_dict, n_neurons=40):
    all_states = [trials_dict[t]['ready_set'].reshape(-1, n_neurons) for t in [800]] + [trials_dict[t]['set_end'].reshape(-1, n_neurons) for t in trials_dict.keys()]
    scaler = StandardScaler()
    scaler.fit(np.vstack(all_states))
    for t in trials_dict.keys():
        for trial in range(len(trials_dict[t]['ready_set'])):
            trials_dict[t]['ready_set'][trial] = scaler.transform(trials_dict[t]['ready_set'][trial])
            trials_dict[t]['set_end'][trial] = scaler.transform(trials_dict[t]['set_end'][trial])


def plot_graph_from_experiments_rsg():
    ts_max = 800
    set_edge = 2
    trials_dict = load_pickle('data/rsg_data_dict.pkl')
    ts_to_set_index = {ts: len(trials_dict[ts]['ready_set'][0])-1 for ts in trials_dict.keys()}
    ts_to_go_index = {ts: len(trials_dict[ts]['set_go'][0]) for ts in trials_dict.keys()}
    scale_rsg_data(trials_dict)
    ready_set_max = trials_dict[ts_max]['ready_set']

    #construct graph
    G = create_lace_from_dynamics(ready_set_max, np.zeros((len(ready_set_max[0]), 1)))
    for ts in trials_dict.keys():
        output = np.zeros((len(trials_dict[ts]['set_end'][0]), 1))
        output[ts_to_go_index[ts]:] = 1
        lace = create_lace_from_dynamics(trials_dict[ts]['set_end'], output)
        #connect ready-set trajectory to set-end via an input edge
        G, start_node = connect_lace_to_graph(G, lace, ts_to_set_index[ts], set_edge)

    original_G = G

    # define rule-set
    remove_zero_paths = lambda G: remove_uninformative_paths(G, 0)
    remove_output_paths = lambda G: remove_uninformative_paths(G, 1)
    rule_set = [remove_zero_paths, remove_output_paths]
    for threshold in [0.0, 0.1, 0.2, 0.5]:
        G = original_G.copy()
        G = merge_components(G, threshold)
        G = step2(G, rule_set)
        plot_interval_reproduction_graph(G)
        plt.show()


def plot_pca_DMS(num_model):
    x, y, h = load_pickle('data/dms_x.pkl'), load_pickle('data/dms_y.pkl'), load_pickle(
        f'data/dms_h_model{num_model}.pkl')
    delay_period = h[:,30:]
    from sklearn.decomposition import PCA
    pca = PCA(2)
    pca.fit(np.vstack(delay_period))
    fig = plt.figure()
    ax = plt.axes()
    from matplotlib import cm
    colors = cm.Reds(np.linspace(0, 1, 4))
    colors = 8*[colors[1]] + 8*[colors[2]]
    linestyles = 4*['-'] + 8*['--'] + 4*['-']#,'--', '--', '-']
    for i, s in enumerate(delay_period):
        ax.plot(*np.hsplit(pca.transform(s), 2), color=colors[i], linewidth=3, linestyle=linestyles[i])
        ax.scatter(*np.hsplit(pca.transform(s[:1]), 2), s=50, zorder=0, color=colors[i])

    plt.axis('off')
    plt.show()


def plot_delay_match_to_sample(num_model):
    F1_IDX = 25
    F2_IDX = 110
    f_values = [-1, 1, -2, 2]

    x, y, h = load_pickle('data/dms_x.pkl'), load_pickle('data/dms_y.pkl'), load_pickle(
        f'data/dms_h_model{num_model}.pkl')
    G = create_lace_from_dynamics(h[0, :F1_IDX], y[0, :F1_IDX])
    from analysis.graph_utils import get_sink_from_node
    j = 0
    for f1 in f_values:
        delay_period = h[j, F1_IDX+5:F2_IDX]
        lace = create_lace_from_dynamics(delay_period, np.zeros(len(delay_period)))
        G, start_node = connect_lace_to_graph(G, lace, F1_IDX - 1, f1)
        sink = get_sink_from_node(G, start_node)
        for f2 in f_values:
            lace = create_lace_from_dynamics(h[j, F2_IDX:], y[j, F2_IDX:])
            G, start_node = connect_lace_to_graph(G, lace, sink, f2)
            j += 1

    G = merge_components(G)

    # define rule-set
    remove_zero_paths = lambda G: remove_uninformative_paths(G, 0)
    remove_output1_paths = lambda G: remove_uninformative_paths(G, 1)
    remove_output2_paths = lambda G: remove_uninformative_paths(G, 2)
    rule_set = [remove_zero_paths, remove_output1_paths, remove_output2_paths, merge_zero_one_edges]
    G = step2(G, rule_set)
    plot_dms_graph(G)
    plt.show()

def interval_discrimination(num_model):
    l = load_pickle(f'data/interval_discrimination_list_model{num_model}.pkl')
    o_dict, s_dict = l[0], l[1]

    t_min, t_max = 10, 30
    t1_range = 17
    t1_t2_space = 5
    pulse = 5
    G = create_lace_from_dynamics(s_dict[t_max][t_min]['T1'], o_dict[t_max][t_min]['T1'])
    node_before_s1 = 4
    for t1 in [17]:
        time = t1
        k = max(list(o_dict[t1]))
        lace = create_lace_from_dynamics(s_dict[t1][k]['T2'], o_dict[t1][k]['T2'])
        G, start_node = connect_lace_to_graph(G, lace, time - 1, node_before_s1)
        path_t1 = get_path_from_source(G, get_target_of_type(G, time - 1, node_before_s1))
        for t2 in [t1 - t1_t2_space, t1 + t1_t2_space]:
            lace = create_lace_from_dynamics(s_dict[t1][t2]['OUTPUT'][pulse:], o_dict[t1][t2]['OUTPUT'][pulse:])
            G, start_node = connect_lace_to_graph(G, lace, path_t1[t2 - 1], 2)

    G = merge_components(G)
    mark_output_nodes(G)
    # define rule-set
    remove_zero_paths = lambda G: remove_uninformative_paths(G, 0)
    remove_output1_paths = lambda G: remove_uninformative_paths(G, 1)
    remove_output2_paths = lambda G: remove_uninformative_paths(G, -1)
    rule_set = [remove_zero_paths, remove_output1_paths, remove_output2_paths, merge_zero_one_edges, merge_edges]
    G = step2(G, rule_set)
    plot_interval_discrimination_graph(G)
    plt.show()


def interval_reproduction(num_model):
    l = load_pickle(f'data/rsg_list_model{num_model}.pkl')
    o_dict, s_dict = l[0], l[1]
    ts_min, ts_max = 30, 50
    G = create_lace_from_dynamics(s_dict[ts_max]['READY_SET'], o_dict[ts_max]['READY_SET'])
    for t1 in range(ts_min, ts_max + 1, 20):
        time = t1
        k = max(list(o_dict[t1]))
        lace = create_lace_from_dynamics(s_dict[t1]['SET_GO'], o_dict[t1]['SET_GO'])
        G, start_node = connect_lace_to_graph(G, lace, time - 10, 2)

    G = merge_components(G)
    mark_output_nodes_rsg(G)

    # define rule-set
    remove_zero_paths = lambda G: remove_uninformative_paths(G, 0)
    remove_output1_paths = lambda G: remove_uninformative_paths(G, 1)
    rule_set = [remove_zero_paths, remove_output1_paths, merge_zero_one_edges, merge_edges]
    G = step2(G, rule_set)
    plot_interval_reproduction_graph(G)
    plt.show()


plot_delay_match_to_sample(0)
plot_delay_match_to_sample(1)
plot_delay_match_to_sample(2)
interval_reproduction(0)
interval_reproduction(1)
interval_discrimination(0)
interval_discrimination(1)
plot_pca_DMS(0)
plot_pca_DMS(1)
plot_pca_DMS(2)
plot_graph_from_experiments_rsg()