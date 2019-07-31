from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import click
from export_model import C2Exporter

@click.command()
@click.option("--init_net", type=click.Path())
@click.option("--pred_net", type=click.Path())
@click.option("--output", type=click.Path())
@click.option("--version", type=str, default=None)
def main(
    init_net,
    pred_net,
    output,
    version
):
    exporter = C2Exporter(init_net, pred_net, None, output, None)
    exporter.set_version(version)
    exporter.save_model(output + '/' + exporter.init_net_path.split("/")[-1], exporter.init_net)
    exporter.save_model(output + '/' + exporter.pred_net_path.split("/")[-1], exporter.pred_net)

if __name__ == "__main__":
    main()
