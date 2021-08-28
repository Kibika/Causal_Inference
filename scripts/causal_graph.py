from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure.notears import from_pandas, from_pandas_lasso
# import pygraphviz





def graph(df):
    sm = from_pandas(df, w_threshold=0.8)
    viz = plot_structure(
        sm,
        graph_attributes={"scale": "2.0", 'size': 2.5},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK)
    Image(viz.draw(format='png'))

def graph_lasso(df):
    sm = from_pandas_lasso(df, w_threshold=0.8, beta=0.8)
    viz = plot_structure(
        sm,
        graph_attributes={"scale": "2.0", 'size': 2.5},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK)
    Image(viz.draw(format='png'))


# constrain output
def graph_constrained(df):
    sm = from_pandas(df, tabu_parent_nodes=['diagnosis'], w_threshold=0.8)
    viz = plot_structure(
        sm,
        graph_attributes={"scale": "2.0", 'size': 2.5},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK)
    Image(viz.draw(format='png'))


def graph_lasso_constrained(df):
    sm = from_pandas_lasso(df, tabu_parent_nodes=['diagnosis'], w_threshold=0.8, beta=0.8)
    viz = plot_structure(
        sm,
        graph_attributes={"scale": "2.0", 'size': 2.5},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK)
    Image(viz.draw(format='png'))




