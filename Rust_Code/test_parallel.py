import numpy as np
import raybnn_python


def test_rows_dot():
    x = np.ones((128, 1024), dtype=np.float64)
    y = np.ones((1024,), dtype=np.float64)
    z = raybnn_python.rows_dot(x, y)
    np.testing.assert_array_almost_equal(z, 1024 * np.ones((128,), dtype=np.float64))
