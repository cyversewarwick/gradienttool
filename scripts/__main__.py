from __future__ import absolute_import, division, print_function

import sys
import optparse
import csv
import re

import numpy as np
import scipy as sp

import shared as ip2
import gradienttool as gt

def cmdparser(args):
    op = optparse.OptionParser()
    op.set_usage("usage: %prog [options] FILE.csv")
    op.set_defaults(verbose=False,normalise=True)
    op.add_option('-v','--verbose',dest='verbose',action='store_true',
                  help="display verbose output")
    op.add_option('-o','--csvoutput',dest='csvoutput',
                  help="write CSV output to FILE", metavar='FILE')
    op.add_option('-p','--pdfoutput',dest='pdfoutput',
                  help="write PDF output to FILE", metavar='FILE')
    op.add_option('-N','--normalise',dest='normalise',action='store_true',
                  help="normalise the data before performing inference")
    op.add_option('-n','--nonormalise',dest='normalise',action='store_false',
                  help="  input is already normalised")
    return op.parse_args(args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    (op,fname) = cmdparser(args)

    if len(fname) == 0:
        sys.stderr.write("Error: Please specify the filename to process, or run with '-h' for more options\n")
        sys.exit(1)

    if len(fname) != 1:
        sys.stderr.write("Error: Only one input filename currently supported\n")
        sys.exit(1)

    inp = ip2.readCsvNamedMatrix(open(fname[0]))

    # assume column headers are time, so remove anything that isn't
    # part of a number and convert into a NumPy matrix
    time = np.asarray([re.sub('[^0-9.]','',x) for x in inp.colnames],
                      dtype=np.float64)

    if op.normalise:
        xtrans = (min(time), max(time)-min(time))
        ytrans = (np.mean(inp.data), np.std(inp.data))
        gradtool = lambda Y: gt.GradientToolNormalised(time, Y, xtrans, ytrans)
    else:
        gradtool = lambda Y: gt.GradientTool(time, Y)

    gl = []
    zeros = {}

    if op.csvoutput is None:
        csvout = csv.writer(sys.stdout)
    else:
        csvout = csv.writer(open(op.csvoutput,'w'))

    csvout.writerow(['item','X','f_mean','f_variance','df_mean','df_variance','tscore'])
    for i in range(inp.nrows()):
        # run the gradient tool
        g = gradtool(inp.data[i,:])
        g.setPriorRbfLengthscale(2.0, 0.2)
        g.setPriorRbfVariance(2.0, 0.5)
        g.setPriorNoiseVariance(1.5, 0.1)
        g.optimize()

        # save results for plotting
        if op.pdfoutput is not None:
            gl.append(g)

        # get the "results" out
        res = g.getResults()

        zt = gt.zerosLinearInterpolate(np.vstack([res.index,res["mud"]]).T)
        if len(zt) in zeros:
            zeros[len(zt)] = np.concatenate([zeros[len(zt)], zt])
        else:
            zeros[len(zt)] = zt

        # write out to CSV file
        name = inp.rownames[i]
        for i,k in res.iterrows():
            csvout.writerow([name,i]+k.tolist())

    # generate PDF output if requested
    if op.pdfoutput is not None:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        import seaborn as sbs

        # write out a nicely ordered PDF file
        plt.switch_backend('Agg')
        with PdfPages(op.pdfoutput) as pdf:
            for a,b in zeros.items():
                fig = plt.figure(figsize=(6,4))
                ax1 = fig.add_subplot(111)
                ax1.margins(0.04)
                ax1.hist(b, edgecolor='none')
                ax1.set_title("%i Gradient Inflections" % (a,))
                pdf.savefig()
                plt.close(fig)

            for i in np.argsort([-g.rbfLengthscale for g in gl]):
                g = gl[i]

                fig = plt.figure(figsize=(8,7))
                gt.doReportPlot(fig, g, title=inp.rownames[i])
                fig.tight_layout()
                pdf.savefig()
                plt.close(fig)

if __name__ == '__main__':
    main()
