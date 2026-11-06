from typing import cast

from scipy.sparse import csr_array, sparray, spmatrix


def differing_entries(left: sparray | spmatrix, right: sparray | spmatrix) -> int:
    """Count entries where two sparse matrices differ (0 means equal).

    SciPy's type stubs declare ``left != right`` as returning ``bool``, but for
    sparse matrices it returns a sparse boolean matrix of the differences.
    """
    return cast(csr_array, left != right).nnz
