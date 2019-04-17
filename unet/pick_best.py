from tvm.autotvm.record import pick_best

import click
import os

@click.command()
@click.option('--in_files', '-i', required=True, multiple=True, type=str)
@click.option('--out_file', '-o', default="autotvm_best.log", type=str)
def pick_best_from_files(in_files, out_file):
    if len(in_files) > 1:
        tmp_file = out_file + '.tmp'
        fout = open(tmp_file, 'w')
        for in_file in in_files:
            best = in_file + '.best.log'
            pick_best(in_file, best)
            with open(best, 'r') as fin:
                s = fin.read()
                fout.write(s)
            os.remove(best)
        fout.close()
        pick_best(tmp_file, out_file)
        os.remove(tmp_file)
    else:
        pick_best(in_files[0], out_file)

if __name__ == '__main__':
    pick_best_from_files()
