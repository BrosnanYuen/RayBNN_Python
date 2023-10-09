import numpy as np
import raybnn_python






x = np.ones((128, 1024), dtype=np.float64)
y = np.ones((1024,), dtype=np.float64)

print(x)

z = raybnn_python.rows_dot(x, y)

print(z)



