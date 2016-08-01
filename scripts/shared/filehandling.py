import csv

import numpy as np

class NamedMatrix(object):
    """Bit like a Pandas style DataFrame, but just a matrix with named
    columns and rows.

    Would this be useful as a subclass of numpy.ndarray?"""
    def __init__(self, colnames, rownames, data, topleft=""):
        assert data.shape == (len(rownames),len(colnames))
        self.topleft  = topleft
        self.colnames = colnames
        self.rownames = rownames
        self.data     = data

    def nrows(self): return len(self.rownames)
    def ncols(self): return len(self.colnames)

    def writeCsv(self, fd):
        out = csv.writer(fd)
        out.writerow([self.topleft]+self.colnames)
        for i in range(len(self.rownames)):
            out.writerow([self.rownames[i]]+self.data[i,:].tolist())

def readCsvNamedMatrix(itr, dtype=np.float64):
    "Read a matrix that has a single row&column of names."
    itr = csv.reader(itr)
    hdr = next(itr)
    rows = []
    data = []

    for l in itr:
        rows.append(l[0])
        data.append(l[1:])

    return NamedMatrix(hdr[1:], rows, np.asarray(data, dtype=dtype), hdr[0])

def _test():
    import numpy.testing as npt
    import StringIO

    # create a dummy string
    inp = ",a,b\r\nx,1.0,2.0\r\ny,5.0,6.0\r\nz,8.0,9.0\r\n"

    # parse CSV
    x = readCsvNamedMatrix(inp.splitlines())

    # make sure we got what we expected
    assert(x.colnames == ["a","b"])
    assert(x.rownames == ["x","y","z"])
    npt.assert_almost_equal(x.data, np.array([[1,2],[5,6],[8,9]]))

    # convert back to CSV format
    buf = StringIO.StringIO()
    x.writeCsv(buf)

    # make sure it looks the same (input must be carefully constructed
    # to ensure this!)
    assert (inp == buf.getvalue())

if __name__ == '__main__':
    _test()
