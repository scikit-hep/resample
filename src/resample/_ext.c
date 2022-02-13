#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

int rcond2(double*, int, const double*, int, const double*);

static PyObject* rcond2_wrap(PyObject *self, PyObject *args)
{
  PyObject *m = NULL, *r = NULL, *c = NULL;
  PyArrayObject *ma = NULL, *ra = NULL, *ca = NULL;

  if(!PyArg_ParseTuple(args, "O!O!O!",
    &PyArray_Type, &m,
    &PyArray_Type, &r,
    &PyArray_Type, &c))
      return NULL;

  ma = (PyArrayObject*)PyArray_FROM_OTF(m, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
  if (!ma) return NULL;
  ra = (PyArrayObject*)PyArray_FROM_OTF(r, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!ra) goto fail;
  ca = (PyArrayObject*)PyArray_FROM_OTF(c, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!ca) goto fail;

  if (PyArray_NDIM(ma) != 2) goto fail;
  if (PyArray_NDIM(ra) != 1) goto fail;
  if (PyArray_NDIM(ca) != 1) goto fail;

  npy_intp* m_shape = PyArray_DIMS(ma);
  npy_intp* r_shape = PyArray_DIMS(ra);
  npy_intp* c_shape = PyArray_DIMS(ca);

  if (m_shape[0] != r_shape[0] || m_shape[1] != c_shape[0]) {
    PyErr_SetString(PyExc_RuntimeError, "shapes do not match");
    goto fail;
  }

  if (rcond2((double*)PyArray_DATA(ma),
             c_shape[0], (const double*)PyArray_DATA(ca),
             r_shape[0], (const double*)PyArray_DATA(ra)) != 0) {
    PyErr_SetString(PyExc_RuntimeError, "error in rcond");
    goto fail;
  }

  // clean up
  Py_DECREF(ra);
  Py_DECREF(ca);
  Py_DECREF(ma);

  Py_RETURN_NONE;

fail:
  Py_XDECREF(ra);
  Py_XDECREF(ca);
  Py_XDECREF(ma);
  return NULL;
}

static PyMethodDef methods[] = {
    {"rcond2", rcond2_wrap, METH_VARARGS},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mod = {
  PyModuleDef_HEAD_INIT,
  "resample._ext",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC
PyInit__ext(void)
{
  PyObject* m = PyModule_Create(&mod);
  if (!m)
    return NULL;
  import_array();
  return m;
}
