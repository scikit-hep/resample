#include <math.h>

double lfac(double x) { return lgamma(x + 1.0); }

double* ptr(double* m, int nr, int nc, int ir, int ic) {
  return m + nc * ir;
}

// Patefield algorithm as 159 Appl. Statist. (1981) vol. 30, no. 1
int rcond2(double* matrix, int nr, const double* r, int nc, const double* c) {
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
      const double ie = ic; // eigth term
      ic -= id;
      ib = ie - ia;
      const double ii = ib - id; // forth term
      if (ie == 0) {
        for (int j = m; j < nc - 1; ++j)
          *ptr(matrix, nr, nc, l, j) = 0;
        ia = 0;
        break;
      }
      double z = 0.5; // TODO replace 0.5 with uniform number generation
      double nlm;
      l131: nlm = ia * id / ie + 0.5;
      double x = exp(
          lfac(ia)
          + lfac(ib) // second?
          + lfac(ic) // second?
          + lfac(id) // third
          - lfac(nlm)
          - lfac(id - nlm)
          - lfac(ia - nlm)
          - lfac(ie)
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
      l150:
      if (lsm) goto l155;
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
      z = 0.5 * sumprb; // TODO replace 0.5 with uniform number generation
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
