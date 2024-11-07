"""
The "core"-library containing wrapper and helper functions. It is intended to
outsource some of the logic of the main library to make the used interface
classes at least a little bit more readable and smaller. They get large enough
anyway, though...
"""

from .pandas import (
    dataframes_equal,
    reorder_column,
    merge_columns,
    merge_and_clip,
    unite_str_columns,
    pivot_to_wide,
    append_missing_text_cols,
    append_mc_columns,
)

from .scores import (
    score_mc_items,
    construct_kprim_threshold,
    apply_kprim_threshold,
    score_mc_tasks,
    drop_missing_threshold,
    drop_earlystopping_threshold,
)

from .batched_iter import (
    BatchIter,
    batched,
)

from .validation import (
    check_key,
    check_options,
    check_type,
    check_uniqueness,
)

from .classifier import (
    print_torch_param_count,
    Classifier,
    ScikitClassifier,
    PyTorchClassifier,
    DenseNN,
)

from .text import (
    cleanup_whitespaces,
    load_abbreviation_regex,
    replace_abbreviations,
    set_item_task_prefix,
    check_seq_lens,
)

from .ols import LinearModel

from .sankey_plotly import plot_sankey, plot_sankey_wrapper

from .io import read_data, write_data
