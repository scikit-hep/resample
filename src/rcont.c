#include <math.h>
#include <numpy/random/distributions.h>

double lfac(double x) { return lgamma(x + 1.0); }

double* ptr(double* m, int nr, int nc, int ir, int ic) {
  return m + nc * ir + ic;
}

/*
  Generate random two-way table with given marginal totals.

  A naive shuffling algorithm has O(N) complexity in space and time, where N is
  the total number of entries in the input array. This algorithm has O(K)
  complexity in time and requires no extra space, where K is the total number of
  cells in the table. For N >> K, which can easily happen in high-energy physics,
  the latter will be dramatically faster.

  Patefield's algorithm adapted from AS 159 Appl. Statist. (1981) vol. 30, no. 1.

  The original FORTRAN code was hand-translated to C. Changes:

  - The section that computed a look-up table of log-factorials was replaced with
    calls to lgamma, which lifts the limitation that the code only works for tables
    with less then 5000 entries, although the algorithm will still be slow for very
    large tables.
  - The data type of input and output arrays was changed to double to minimize
    type conversions.
  - The original implementation allocated a column vector JWORK, but this is not
    necessary. One can use the last column of the output matrix as work space.
  - The function uses Numpy's random number generator and distribution library.
  - Checks for zero entries in row and column vector were integrated into the main
    algorithm.
*/
int rcont(double* matrix, int nr, const double* r, int nc, const double* c,
          double ntot, bitgen_t* bitgen_state) {
  if (matrix == 0)
    return 1;

  if (nr < 2)
    return 2;

  if (nc < 2)
    return 3;

  // jwork can be folded into matrix using last row
  double* jwork = ptr(matrix, nr, nc, nr - 1, 0);
  double jc = 0;
  for (int i = 0; i < nc; ++i) {
    if (c[i] == 0) // no zero entries allowed in c
      return 5;
    jwork[i] = c[i];
    jc += c[i];
  }
  if (jc == 0) // ntotal must be positive
    return 4;
  if (r[nr - 1] == 0) // no zero entries allowed in r
    return 6;
  double ib = 0;
  // last row is not random due to constraint
  for (int l = 0; l < nr - 1; ++l) {
    double ia = r[l]; // first term
    if (ia == 0) // no zero entries allowed in r
      return 6;
    double ic = jc; // second term
    jc -= r[l];
    // last column is not random due to constraint
    for (int m = 0; m < nc - 1; ++m) {
      const double id = jwork[m]; // third term
      const double ie = ic; // eight term
      ic -= id;
      ib = ie - ia;
      const double ii = ib - id; // forth term
      if (ie == 0) {
        for (int j = m; j < nc - 1; ++j)
          *ptr(matrix, nr, nc, l, j) = 0;
        ia = 0;
        break;
      }
      double z = random_standard_uniform(bitgen_state);
      double nlm;
      l131: nlm = floor(ia * id / ie + 0.5);
      double x = exp(
          lfac(ia)
          + lfac(ib)
          + lfac(ic)
          + lfac(id)
          - lfac(ie)
          - lfac(nlm)
          - lfac(id - nlm)
          - lfac(ia - nlm)
          - lfac(ii + nlm)
      );
      if (x >= z) goto l160;
      double sumprb = x;
      double y = x;
      double nll = nlm;
      int lsp = 0;
      int lsm = 0;
      // increment entry at (l,m)
      double j;
      l140: j = (id - nlm) * (ia - nlm);
      if (j == 0) goto l156;
      nlm += 1;
      x *= j / (nlm * (ii + nlm));
      sumprb += x;
      if (sumprb >= z) goto l160;
      l150: if (lsm) goto l155;
      // decrement entry at (l,m)
      j = nll * (ii + nll);
      if (j == 0) goto l154;
      nll -= 1;
      y *= j / ((id - nll) * (ia - nll));
      sumprb += y;
      if (sumprb >= z) goto l159;
      if (!lsp) goto l140;
      goto l150;
      l154: lsm = 1;
      l155: if (!lsp) goto l140;
      z = random_standard_uniform(bitgen_state) * sumprb;
      goto l131;
      l156: lsp = 1;
      goto l150;
      l159: nlm = nll;
      l160: *ptr(matrix, nr, nc, l, m) = nlm;
      ia -= nlm;
      jwork[m] -= nlm;
    }
    // compute entry in last column of matrix
    *ptr(matrix, nr, nc, l, nc-1) = ia;
  }
  // compute entries in last row of matrix
  // jwork is already last row of matrix, so nothing to be done up to nc - 2
  *ptr(matrix, nr, nc, nr-1, nc-1) = ib - *ptr(matrix, nr, nc, nr - 1, nc - 2);

  return 0;
}
