import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure as plotlyfig
from seaborn import color_palette
from matplotlib.colors import to_rgba




def df_to_sankey_data(
        df: pd.DataFrame,
        cols: list[str],
        label_dict: dict=None,
    ) -> tuple[list, list, list, list]:
    """ Wrapper to transform longitudinal or longitudinal-like data from a
    representation in a dataframe with one column for each stage
    (e. g. measurement time) and one row per instance into tuples that fit into
    plotly's sankey diagram plot syntax.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with one column for each stage (e. g. measurement time) and
        one row per instance.
    cols : list[str]
        List with the column names of the stage-columns.
    label_dict : dict=None
        Dictionary to map the sankey labels to other labels. If None, the labels
        are the unique values in the columns.

    Returns
    -------
    tuple[list, list, list, list]
        source, target, value, labels lists in that order.
    """

    assert all(col in df.columns for col in cols), (
        "Columns given are not all in df.columns."
    )

    n_transitions = len(cols)-1

    # Extracting the levels from the data
    levels = []
    num_levels = [] # Sankey works with one numerical label per group AND stage.
    for col in cols:
        levels.append(np.sort(np.unique(df[col])))
        num_levels.append(np.arange(levels[-1].size))
    for k in range(len(num_levels)):
        if k == 0:
            continue
        else:
            for i in range(k):
                num_levels[k] += num_levels[i].size

    # Defining the source list based on the extracted levels. There is one
    # source-entry of each "previous"-level per "post"-level
    source = []
    for k in range(len(num_levels)-1):
        source += np.repeat(num_levels[k], repeats=len(num_levels[k+1])).tolist()

    # Constructing a dict for later counting in the columns
    labels_map = {}
    for k in range(len(levels)):
        labels_map.update(dict(zip(num_levels[k], levels[k])))

    # Definint the target list based on the source list
    target = []
    for k in range(1, len(num_levels)):
        for i in range(len(num_levels[k-1])):
            target += num_levels[k].tolist()

    # Extracting the transition values
    value = []
    npsource = np.array(source)
    nptarget = np.array(target)
    for n in range(n_transitions):
        max1 = num_levels[n].max()
        min1 = num_levels[n].min()
        max2 = num_levels[n+1].max()
        min2 = num_levels[n+1].min()

        for pre, post in zip(
            npsource[np.logical_and(npsource <= max1, npsource >= min1)],
            nptarget[np.logical_and(nptarget <= max2, nptarget >= min2)]
        ):
            pre_mapped = labels_map[pre]
            post_mapped = labels_map[post]
            value.append(
                df.loc[(df[cols[n]]==pre_mapped) & (df[cols[n+1]]==post_mapped), :].shape[0]  # noqa: E501
            )

    # Defining labels
    labels = []
    for k in range(len(levels)):
        for x in levels[k]:
            lbl = f"Group {x}"
            if label_dict is not None:
                lbl = label_dict.get(x, lbl)
            labels.append(lbl)

    # Omit connections from dropout to dropout
    for idx, (tar, src) in enumerate(zip(target, source)):
        if labels_map[tar] == 99 and labels_map[src] == 99:
            value[idx] = 0

    return source, target, value, labels, labels_map




def plot_sankey(
        source: list[int],
        target: list[int],
        value: list[int],
        label: list[str],
        node_colors: list[str],
        link_colors: list[str],
        title: str="title",
    ) -> plotlyfig:
    """Wrapper to plot a sankey plot for longitudinal (like) data.
    It returns the plotly figure as an object.

    Parameters
    ----------
    source : list[int]
        List with the source integers. There must be one integer per stage AND
        group, with the last stage omitted.
    target : list[int]
        List with the target integers. There must be one integer per stage AND
        group, with the first stage omitted.
    value : list[int]
        List with the tranision values from source to target.
    label : list[str]
        List with the labels for the integers in source and stage
    node_colors : list[str]
        Colors for the nodes and links.
    link_colors : list[str]
        Colors for the nodes and links.
    title : str="title"
        Title of the plot


    Returns
    -------
    plotly.graph_objs._figure.Figure
    """

    source = list(source)
    target = list(target)
    value = list(value)

    assert len(source) == len(target) and len(source) == len(value), (
        "`source`, `target` and `value` must be of the same length."
    )

    assert len(set(source+target)) == len(label), (
        "There must be a `label` for every unique integer in the union of" +
        "`source` and `target`."
    )

    fig = go.Figure(data=[go.Sankey(
        # arrangement='snap',
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = label,
            color = node_colors
        ),
        link = dict(
            source = source,
            target = target,
            color = link_colors,
            value = value,
        )
    )])

    fig.update_layout(title_text=title, font_size=10)

    return fig




def plot_sankey_wrapper(
        df: pd.DataFrame,
        cols: list[str],
        label_dict: dict=None,
        dropout_as_source: bool=True,
        dropout_as_target: bool=True,
        title: str="Sankey Plot of Clusters",
        palette = None,
    ) -> plotlyfig:
    """Wrapper for directly creating sankey plots with plotly out of dataframes
    with one column per timestamp and one row per instance. It mainly
    combines [df_to_sankey_data][qutools.core.sankey_plotly.df_to_sankey_data]
    and [plot_sankey][qutools.core.sankey_plotly.plot_sankey] but adds automatic
    color generations and options wether to (not) inlude dropout in different
    ways.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing
        $n_{\\textsf{instances}} \\times n_{\\textsf{timestamps}}$
    cols : list[str]
        The timestamp columns.
    label_dict : dict=None
        Dictionary to map the sankey labels to other labels. If None, the labels
        are the unique values in the columns.
    dropout_as_source : bool=True
        Wether dropout should be used as source. Note that the groupsizes
        depend on in- and outstreaming amounts in sankey plots so excluding
        dropout alters these sizes. If dropout is excluded as source,
        persons first occuring e. g. in the second timestamp will not be
        included.
    dropout_as_target : bool=True
        Wether dropout should be used as target. Note that the groupsizes
        depend on in- and outstreaming amounts in sankey plots so excluding
        dropout alters these sizes.
    title : str="Sankey Plot of Clusters"
        A title.
    palette = None
        A palette for the clusters to overwrite the automatically set hls.

    Returns
    -------
    plotly.graph_objs._figure.Figure
    """
    source, target, value, labels, labels_map = df_to_sankey_data(
        df=df,
        cols=cols,
        label_dict=label_dict
    )
    labels = [label.replace("Group 99", "Dropout") for label in labels]
    has_dropout = 99 in labels_map.values()

    if has_dropout:
        n_cluster = len(set(labels_map.values())) - 1
    else:
        n_cluster = len(set(labels_map.values()))

    # Setting specific values to 0 dependent on dropout-handling
    if not dropout_as_source:
        for idx, src in enumerate(source):
            if labels_map[src] == 99:
                value[idx] = 0
    if not dropout_as_target:
        for idx, tar in enumerate(target):
            if labels_map[tar] == 99:
                value[idx] = 0

    # Colors
    if palette is None:
        palette = list(color_palette("hls", n_cluster).as_hex())
    if has_dropout:
        palette.append("#FFFFFF")
    node_colors = list(np.array(3 * palette).flatten())

    link_colors = []
    palette = [
        f"rgba{to_rgba(c, alpha=0.5)}"
        for c in color_palette("hls", n_cluster, desat=0.5)
    ]
    if has_dropout:
        palette.append(f"rgba{to_rgba('#dddddd', alpha=0.3)}")

    for tar, src in zip(target, source):
        tar_label = labels_map[tar]
        src_label = labels_map[src]
        if tar_label == 99 or src_label == 99:
            link_colors.append(palette[-1])
        else:
            link_colors.append(palette[int(src_label-1)])

    p = plot_sankey(
        source, target, value, labels,
        node_colors=node_colors,
        link_colors=link_colors,
        title=title
    )

    return p
