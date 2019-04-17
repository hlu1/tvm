import click
import nnvm
import json

def load_graph(f):
    graph = json.load(f)
    # print(graph)
    return graph

@click.command()
@click.option('--graph', required=True, type=str)
@click.option('--dst', default=None, type=str)
def run(graph, dst):
    with open(graph, 'rb') as f:
        jgraph = load_graph(f)
    if dst is not None:
        with open(dst, 'w') as f:
            f.write(jgraph)
    import ipdb; ipdb.set_trace()
    nodes = jgraph['nodes']
    shapes = jgraph['attrs']['shape'][1]
    dltype = jgraph['attrs']['dltype'][1]
    graph = nnvm.graph.load_json(jgraph)

    print(graph)


if __name__ == '__main__':
    run()
