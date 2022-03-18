import seaborn as sns

DISTANCE_THRESHOLD = 1
OUTPUT_THRESHOLD = 0.5
sns.set()
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from analysis.compress_graph import get_zero_paths, get_path_from_source
from analysis.graph_utils import get_state_output_from_path, get_zero_graph, get_property_from_path, connect_cycle, create_lace, get_available_node, shift_nodes_names
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import networkx as nx

def find_merge_indices(states1, outputs1, states2, outputs2):
    if np.abs(np.max(outputs1) - np.max(outputs2)) > 0.5:
        raise NameError
    if min(np.max(outputs1), np.max(outputs2)) < 1:
        mat = new_state_distance_matrix(states2, states1)
        mat_output = max_bottom_left_diagonal(distance_matrix(outputs2, outputs1))
        path2_idx, path1_idx, min_val = calculate_merge_index(mat)
        while mat_output[path2_idx, path1_idx] > 0.1:
            path1_idx, path2_idx = path1_idx + 1, path2_idx + 1
            if path2_idx == len(states2) or path1_idx == len(states1):
                raise NameError
    else:
        ii, jj, _ = calculate_merge_index(max_bottom_left_diagonal(distance_matrix(outputs2, outputs1)))
        mat = new_state_distance_matrix(states2, states1, k=jj - ii)
        path2_idx, path1_idx, min_val = calculate_merge_index_diagonal(max_bottom_left_diagonal(mat), k=jj - ii)

    return path1_idx, path2_idx


def try_find_merge_location_for_components(c1, c2):
    paths_c1 = get_zero_paths(c1)[0]
    paths_c2 = get_zero_paths(c2)[0]

    compatible_paths = {}
    for idx1, path1 in enumerate(paths_c1):
        path1_states, path1_outputs = get_state_output_from_path(c1, path1)
        compatible_paths[idx1] = None
        min_index = np.inf
        for idx2, path2 in enumerate(paths_c2):
            path2_states, path2_outputs = get_state_output_from_path(c2, path2)
            try:
                path1_idx, path2_idx = find_merge_indices(path1_states, path1_outputs, path2_states, path2_outputs)
                if path1_idx < min_index:
                    compatible_paths[idx1] = (len(path1) - path1_idx, idx2, len(path2) - path2_idx)
                    min_index = path1_idx

            except NameError:
                continue

    triplets = list(compatible_paths.values())
    merge_size = [min(t[0], t[2]) if t is not None else 0 for t in triplets]
    best_path1 = np.argmax(merge_size)
    if merge_size[best_path1] > 0:
        t = triplets[best_path1]
        node1 = paths_c1[best_path1][-t[0]]
        node2 = paths_c2[t[1]][-t[2]]
        return node1, node2

    return -1, -1


def merge_two_paths(G, path1, path2):
    if len(path1) < len(path2):
        path1, path2 = path2, path1
    for j in range(min(len(path2), len(path1))):
        G = nx.contracted_nodes(G, path1[j], path2[j])

    s = path1[j]
    targets_zero = sorted([t for t in G.successors(s) if G[s][t]['edge_type'] == 0])
    while len(targets_zero) > 1:
        G = nx.contracted_nodes(G, targets_zero[0], targets_zero[1])
        s = targets_zero[0]
        targets_zero = sorted([t for t in G.successors(s) if G[s][t]['edge_type'] == 0])

    return G


def merge_components(G, threshold):
    global DISTANCE_THRESHOLD
    DISTANCE_THRESHOLD = threshold
    while True:
        G_zero = get_zero_graph(G)
        components = [nx.subgraph(G_zero, c) for c in nx.weakly_connected_components(G_zero)]
        import itertools
        pairs = list(itertools.combinations(components[1:], 2))
        for (c1, c2) in pairs[::-1]:
            node1, node2 = try_find_merge_location_for_components(c1, c2)
            if node1 != -1:
                path1 = get_path_from_source(c1, node1)
                path2 = get_path_from_source(c2, node2)
                G = merge_two_paths(G, path1, path2)
                break

        else:
            return G


def generate_input(n, steps):
    input = np.zeros((n, steps, 2))
    input[:,:5, 1] = 1
    return input


def distance_diagonal(a, TS):
    a_norm = np.linalg.norm(a[:-1] - a[1:], axis=1)
    a_norm = np.append(a_norm, a_norm[-1])
    for k in TS:
        if k <= 0:
            continue
        for i in range(max(len(a)-200, 0), len(a) - k):
            for j in range(i + k, len(a), k):
                d = np.linalg.norm(a[i] - a[j])
                if d < 1e-5 or (d < 1 and 2*d/(a_norm[i] + a_norm[j]) < 1 and abs(np.log10(a_norm[i]) - np.log10(a_norm[j])) < 1):
                    continue
                else:
                    break
            else:
                return (i + k, i)

    raise NameError


def max_bottom_left_diagonal(mat):
    n, m = mat.shape
    for i in range(n - 2, -1, -1):
        for j in range(m - 2, -1, -1):
            mat[i, j] = max(mat[i, j], mat[i + 1, j + 1])

    return mat


def speed_vector(vec):
    vec_norm = np.linalg.norm(vec[:-1] - vec[1:], axis=1)
    vec_norm = np.append(vec_norm, vec_norm[-1])
    return vec_norm


def get_discriminant_score(X, y):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X, y)
    return np.sum(np.abs(clf.predict(X) - y))/len(y)


def calculate_discrimination_score(a, b, pca=None):
    if pca is None:
        pca = PCA(5)
        pca.fit(np.vstack([a, b]))
    a, b = pca.transform(a), pca.transform(b)
    X = np.vstack([a, b])
    y = np.array(len(a) * [0] + len(b) * [1])
    score = get_discriminant_score(X, y)
    reps = 10
    score_random = 0
    for _ in range(reps):
        y = np.array(len(a) * [0] + len(b) * [1])
        y = np.random.permutation(y)
        score_random += get_discriminant_score(X, y)

    return score_random/reps - score


def states_distance_matrix(a, b):
    a_norm, b_norm = speed_vector(a), speed_vector(b)
    n, m = len(a), len(b)
    mat = 1000*np.ones((n, m))
    for k in range(-n-1, m):
        for j in range(m - 1, max(k - 1,-1), -1):
            i = j - k
            if i >= n:
                continue

            d = np.linalg.norm(a[i] - b[j])
            if d > 1e-3:
                mat[i, j] = 2*d/(a_norm[i] + b_norm[j])
                if mat[i, j] > 100 or abs(np.log10(a_norm[i]+ 1e-10) - np.log10(b_norm[j] + 1e-10)) > 1:
                    continue#break
            else:
                mat[i, j] = 0

    mat = max_bottom_left_diagonal(mat)
    return mat


def normalized_distance(ai, bj, qai, qbj):
    d = np.linalg.norm(ai - bj)
    if d > 1e-3:
        d = 2 * d / (qai + qbj)
        if d > 100 or abs(np.log10(qai + 1e-10) - np.log10(qbj + 1e-10)) > 1:
            return -1
        return d
    else:
        return 0


def update_distance_in_diagonal(sl1, sl2, k, matrix, pca):
    for j in range(len(sl2) - 1, max(k - 1, -1), -1):
        i = j - k
        if i >= len(sl1):
            continue

        d = calculate_discrimination_score(sl1[i], sl2[j], pca)
        if d > DISTANCE_THRESHOLD:
            break
        matrix[i, j] = d

    return matrix


def new_state_distance_matrix(sl1, sl2, k=None):
    a = np.vstack([np.mean(s, axis=0) for s in sl1])
    b = np.vstack([np.mean(s, axis=0) for s in sl2])
    matrix = np.inf*np.ones((len(sl1), len(sl2)))

    sizes = [len(s) for s in sl1] + [len(s) for s in sl2]
    if min(sizes) > 10:
        # use discriminant
        pca = PCA(5)
        pca.fit(np.vstack([np.vstack(sl1), np.vstack(sl2)]))
        if k is None:
            k_range = range(-len(sl1) - 1, len(sl2))
        else:
            k_range = [k]
        for k in k_range:
            matrix = update_distance_in_diagonal(sl1, sl2, k, matrix, pca)
    else:
        # calculate over means
        return states_distance_matrix(a, b)
    return matrix


def diagonal_index_to_matrix_index(location, k):
    if k >= 0:
        row, col = location, location + k
    else:
        row, col = location - k, location

    return row, col


def calculate_merge_index_diagonal(mat, k):
    diagonal = np.array(np.diag(mat, k=k))
    try:
        location = np.argwhere(diagonal < DISTANCE_THRESHOLD).flatten()[0]
        row, col = diagonal_index_to_matrix_index(location, k)
        return row, col, 0
    except:
        raise NameError()



def calculate_merge_index(mat):
    if np.min(mat) >= 50:
        raise NameError()

    n, m = mat.shape[0], mat.shape[1]
    diagonals = []
    min_val = np.inf
    for j in range(m):
        min_row = np.argmin(mat[:,j])
        if mat[min_row, j] < DISTANCE_THRESHOLD:
            min_val = min(mat[min_row, j], min_val)
            diagonals.append(j - min_row)
            if min_val == 0:
                break

    #different for extrapolation vs training, where you don't have long inputs
    from collections import Counter
    most_common_2 = Counter(diagonals).most_common(2)

    if most_common_2 and (most_common_2[0][1] > 3 or min_val < 0.01 or abs(n-m) < 5):
        if len(most_common_2) == 2 and most_common_2[0][1] > 2*most_common_2[1][1]:
            k = most_common_2[0][0]
        else:
            k = diagonals[-1]
    else:
        raise NameError('A very specific bad thing happened')

    diagonal = np.array(np.diag(mat, k=k))
    location = np.argwhere(diagonal < DISTANCE_THRESHOLD).flatten()[0]
    row, col = diagonal_index_to_matrix_index(location, k)
    return row, col, min_val


def merge_paths(G, path_s, path_t):
    path_s = path_s[:len(path_t)]
    for idx in range(len(path_t) - 1, -1, -1):
        G = nx.contracted_nodes(G, path_s[idx], path_t[idx])

    return G

def get_merging_location(G, lace_states, lace_outputs, exclude_node):
    paths = get_zero_paths(G)
    compatible_paths = []#{}

    for component in paths.keys():
        for path in paths[component]:
            if exclude_node == path[0]:
                continue
            path_states = get_property_from_path(G, path, 'state')
            path_outputs = get_property_from_path(G, path, 'output')
            try:
                mat = states_distance_matrix(path_states, lace_states);
                mat_output = max_bottom_left_diagonal(distance_matrix(path_outputs[:], lace_outputs[:], p=1))
                path_idx, lace_idx, min_val = calculate_merge_index(mat)
                while mat_output[path_idx, lace_idx] > OUTPUT_THRESHOLD:
                    path_idx += 1
                    lace_idx += 1
                    if path_idx == len(path_states) or lace_idx == len(lace_states):
                        raise NameError

                compatible_paths.append((lace_idx, path[path_idx:], min_val))

            except NameError:
                continue

    if not bool(compatible_paths):
        lace_offset, path = len(lace_states), []
    else:
        pairs = compatible_paths#list(compatible_paths.values())
        offsets = [p[0] for p in pairs]
        idx = np.argmin(offsets)
        lace_offset, path, _ = pairs[idx]

    return lace_offset, path


def merge_two_cycles(G, cycle1, cycle2):
    cycle1_states = get_property_from_path(G, cycle1, 'state')
    cycle2_states = get_property_from_path(G, cycle2, 'state')
    dists = [np.linalg.norm(cycle2_states[0] - cycle1_states[i]) for i in range(len(cycle1_states))]
    candidate, candidate_val = np.argmin(dists), np.min(dists)
    G.remove_nodes_from(cycle2[1:])
    G = nx.contracted_nodes(G, cycle1[candidate], cycle2[0])
    return G

def svm_if_separable(points1, points2):
    X = np.vstack((points1, points2))
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    X = StandardScaler().fit_transform(X)
    clf = svm.SVC(kernel='linear', C=100, max_iter=1000000)
    y = np.ones(len(points1) + len(points2))
    y[len(points1):] = -1
    clf.fit(X, y)
    score = clf.score(X, y)
    if score == 1:
        width = 2 / np.linalg.norm(clf.coef_)
        return width

    return 0

def merge_cycle_to_graph(G, start_node):
    zero_graph = get_zero_graph(G)
    curr_cycle = [e[0] for e in nx.find_cycle(zero_graph, start_node)]
    curr_cycle_states = get_property_from_path(G, curr_cycle, 'state')
    node = curr_cycle[0]
    all_cycles = nx.simple_cycles(zero_graph)
    all_cycles = [cycle for cycle in all_cycles if node not in cycle and len(cycle) > 1]
    all_cycles = sorted(all_cycles, key=lambda x:x[0])
    all_cycles = [cycle for cycle in all_cycles if 0.8 <= len(cycle) / len(curr_cycle) <= 1.2]
    distance_threshold = 0.1 if len(curr_cycle) >= 10 else 0
    all_cycles = [cycle for cycle in all_cycles if

       svm_if_separable(get_property_from_path(G, cycle, 'state'), curr_cycle_states) <= distance_threshold]
    if all_cycles:
        curr_cycle, all_cycles[0] = all_cycles[0], curr_cycle
        for cycle in all_cycles:
            G = merge_two_cycles(G, curr_cycle, cycle)

    return G


def get_cycle(dynamics):
    dyn1d = PCA(1).fit_transform(dynamics).squeeze()
    corr = np.correlate(dyn1d, dyn1d, mode='full')[:len(dyn1d)]
    corr = corr/np.max(corr)
    corr[corr < 0] = 0
    from scipy.signal import find_peaks
    peaks = find_peaks(corr, prominence=0.05)[0]
    if peaks.size == 0:
        raise NameError
    if peaks.size < 6:
        return [len(dynamics) - 1 - peaks[0]]
    diffs = np.hstack([np.diff(peaks[:]), np.diff(peaks[::2])])
    bins = np.bincount(diffs)
    TS = np.argsort(bins)[-4:]
    TS = [T for T in TS if bins[T] > 1]
    return TS[::-1]


def locate_cycle(dynamics):
    TS = get_cycle(dynamics)
    t, s = distance_diagonal(dynamics, TS)
    return t, s


def create_lace_from_dynamics(dynamics, outputs, try_locate_cycle=False):
    G = create_lace(dynamics, outputs)
    if not try_locate_cycle:
        return G
    try:
        t, s = locate_cycle(dynamics)
        connect_cycle(G, t, s)
    except:
        pass
    return G


def connect_lace_to_graph(G, lace, prev_node, edge_type):
    start_node = get_available_node(G)
    lace = shift_nodes_names(lace, start_node)
    G = nx.union(G, lace)
    if prev_node is not None:
        G.add_edge(prev_node, start_node)
        G[prev_node][start_node]['edge_type'] = edge_type

    return G, start_node


def get_max_peak(array):
    from scipy.signal import find_peaks
    peaks = find_peaks(array, prominence=0.1)[0]
    if len(peaks):
        return peaks[0]

    return -1


def modified_cycle_finder(dynamics, delta):
    if len(dynamics) == delta:
        raise NameError
    a = distance_matrix(dynamics[delta:], dynamics[delta:])
    if len(dynamics) == 2*delta:
        min_diag = 2
    else:
        min_diag = 20

    diagonals = [get_max_peak(-a[i, i + 1:]) + 1 for i in range(len(a))]
    from collections import Counter
    most_commons = Counter(diagonals).most_common(10)
    most_commons = [pair for pair in most_commons if pair[0] > min_diag and pair[1] > 10]
    if most_commons:
        k, count = most_commons[0][0], most_commons[0][1]
        if (len(a) - k) <= 3 * count or len(a) > 400 and count > 30:
            offset = np.argwhere(np.array(diagonals) == k).squeeze()[0] + delta
            return k + offset, offset

    raise NameError
