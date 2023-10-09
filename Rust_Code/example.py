import numpy as np
import raybnn_python






x = np.array([ [-2, 4, 7], [3, 6, 9], [-4, 1, 12]   ], dtype=np.float64)
y = np.array([ [-22, 44], [73, 26]   ], dtype=np.float64)

print(x)

print(y)

print("Rust")

z = raybnn_python.rows_dot(x, y)

print(z)



