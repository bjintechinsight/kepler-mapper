# A small helper class to house functions needed by KeplerMapper.visualize
import numpy as np
from sklearn import preprocessing
import json
from collections import defaultdict


# palette = [
#     '#0500ff', '#0300ff', '#0100ff', '#0002ff', '#0022ff', '#0044ff',
#     '#0064ff', '#0084ff', '#00a4ff', '#00a4ff', '#00c4ff', '#00e4ff',
#     '#00ffd0', '#00ff83', '#00ff36', '#17ff00', '#65ff00', '#b0ff00',
#     '#fdff00', '#FFf000', '#FFdc00', '#FFc800', '#FFb400', '#FFa000',
#     '#FF8c00', '#FF7800', '#FF6400', '#FF5000', '#FF3c00', '#FF2800',
#     '#FF1400', '#FF0000'
# ]

palette = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
    '#808080', '#000000'
]


def format_meta(graph, custom_meta=None):
    n = [l for l in graph["nodes"].values()]
    n_unique = len(set([i for s in n for i in s]))

    if custom_meta is None:
        custom_meta = graph['meta_data']

    mapper_summary = {
        "custom_meta": custom_meta,
        "n_nodes": len(graph["nodes"]),
        "n_edges": sum([len(l) for l in graph["links"].values()]),
        "n_total": sum([len(l) for l in graph["nodes"].values()]),
        "n_unique": n_unique
    }

    return mapper_summary


def format_mapper_data(graph, labels_value, labels_name, X,
                       X_names, lens, lens_names, custom_tooltips, env):
    # import pdb; pdb.set_trace()
    json_dict = {"nodes": [], "links": []}
    node_id_to_num = {}
    for i, (node_id, member_ids) in enumerate(graph["nodes"].items()):
        node_id_to_num[node_id] = i
        # c = _color_function(member_ids, color_function)
        c = _get_color_idx(labels_value[member_ids])
        t = _type_node()
        s = _size_node(member_ids)
        tt = _format_tooltip(env, member_ids, custom_tooltips, X, X_names, lens, lens_names, labels_value, labels_name, node_id)

        n = {"id": "",
             "name": node_id,
             "color": c,
             "type": _type_node(),
             "size": s,
             "tooltip": tt}

        json_dict["nodes"].append(n)
    for i, (node_id, linked_node_ids) in enumerate(graph["links"].items()):
        for linked_node_id in linked_node_ids:
            l = {"source": node_id_to_num[node_id],
                 "target": node_id_to_num[linked_node_id],
                 "width": _size_link_width(graph, node_id, linked_node_id)}
            json_dict["links"].append(l)
    return json_dict

def _get_color(lst):
    idx = _get_color_idx(list(lst))
    return palette[idx]

def _get_color_idx(lst):
    ## get the dominate label, ignore label > 20
    lst_s = [x for x in lst if x < 20 ]
    if lst_s:
        main_label_value = max(set(lst_s), key=lst.count)
    else:
        main_label_value = max(set(lst), key=lst.count)

    return main_label_value

def _format_tooltip(env, member_ids, custom_tooltips, X,
                    X_names, lens, lens_names, labels_value, labels_name, node_ID):
    # TODO: Allow customization in the form of aggregate per node and per entry in node.
    # TODO: Allow users to turn off tooltip completely.

    custom_tooltips = custom_tooltips[member_ids] if custom_tooltips is not None else member_ids

    # list will render better than numpy arrays
    custom_tooltips = list(custom_tooltips)

    projection_stats = _format_projection_statistics(
        member_ids, lens, lens_names)
    cluster_stats = _format_cluster_statistics(member_ids, X, X_names)

    histogram = build_histogram(labels_value[member_ids], labels_value, labels_name)

    tooltip = env.get_template('cluster_tooltip.html').render(
        projection_stats=projection_stats,
        cluster_stats=cluster_stats,
        custom_tooltips=custom_tooltips,
        histogram=histogram,
        dist_label="Member",
        node_id=node_ID)

    return tooltip


def build_histogram(data, labels_value, labels_name):
    # Build histogram of data based on values of labels
    hist = np.histogram(data, range=(min(labels_value), max(labels_value)), bins=len(np.unique(labels_value))) ## one bin per unique value

    histogram = []
    max_bucket_value = max(hist)
    # sum_bucket_value = sum(hist)
    for bar, l_value, l_name in zip(hist, labels_value, labels_name):
        height = int(((bar / max_bucket_value) * 100) + 1)
        # perc = round((bar / sum_bucket_value) * 100., 1)
        color = palette[l_value]

        histogram.append({
            'height': height,
            'label': l_name,
            'color': color
        })
    return histogram


def graph_data_distribution(graph, labels_value, labels_name):

    node_label = []
    for node_id, member_ids in graph["nodes"].items():
        node_label.append(_get_color_idx(labels_value[member_ids]))

    histogram = build_histogram(node_label, labels_value, labels_name)

    return histogram


def _format_cluster_statistics(member_ids, X, X_names):
    # TODO: Cache X_mean and X_std for all clusters.
    # TODO: replace long tuples with named tuples.
    # TODO: Name all the single letter variables.
    # TODO: remove duplication between above_stats and below_stats
    # TODO: Should we only show variables that are much above or below the mean?

    cluster_data = {'above': [], 'below': [], 'size': len(member_ids)}

    cluster_stats = ""
    if X is not None:
        # List vs. numpy handling: cast to numpy array
        if isinstance(X_names, list):
            X_names = np.array(X_names)
        # Defaults when providing no X_names
        if X_names.shape[0] == 0:
            X_names = np.array(["f_%s" % (i) for i in range(
                X.shape[1])])

        cluster_X_mean = np.mean(X[member_ids], axis=0)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        above_mean = cluster_X_mean > X_mean
        std_m = np.sqrt((cluster_X_mean - X_mean)**2) / X_std

        stat_zip = list(zip(std_m, X_names, np.mean(X, axis=0), cluster_X_mean, above_mean, np.std(X, axis=0)))
        stats = sorted(stat_zip, reverse=True)
        above_stats = [a for a in stats if a[4] == True]
        below_stats = [a for a in stats if a[4] == False]

        if len(above_stats) > 0:
            for s, f, i, c, a, v in above_stats[:5]:
                cluster_data['above'].append({
                    'feature': f,
                    'mean': round(c, 3),
                    'std': round(s, 1)
                })

        if len(below_stats) > 0:
            for s, f, i, c, a, v in below_stats[:5]:
                cluster_data['below'].append({
                    'feature': f,
                    'mean': round(c, 3),
                    'std': round(s, 1)
                })

    return cluster_data


def _format_projection_statistics(member_ids, lens, lens_names):
    projection_data = []

    if lens is not None:
        if isinstance(lens_names, list):
            lens_names = np.array(lens_names)

        # Create defaults when providing no lens_names
        if lens_names.shape[0] == 0:
            lens_names = np.array(
                ["p_%s" % (i) for i in range(lens.shape[1])])

        means_v = np.mean(lens[member_ids], axis=0)
        maxs_v = np.max(lens[member_ids], axis=0)
        mins_v = np.min(lens[member_ids], axis=0)

        for name, mean_v, max_v, min_v in zip(lens_names, means_v, maxs_v, mins_v):
            projection_data.append({
                'name': name,
                'mean': round(mean_v, 3),
                'max': round(max_v, 3),
                'min': round(min_v, 3)
            })

    return projection_data



# def _color_function(member_ids, color_function):
#     return _color_idx(np.mean(color_function[member_ids]))
#     # return int(np.mean(color_function[member_ids]) * 30)


# def _color_idx(val):
#     """ Take a value between 0 and 1 and return the idx of color """
#     return int(val * 30)


def _size_node(member_ids):
    # increase the node size for better visualiztion of color
    return int(np.log(len(member_ids) + 1) + 1) + 5


def _type_node():
    return "circle"


def _size_link_width(graph, node_id, linked_node_id):
    return 1
