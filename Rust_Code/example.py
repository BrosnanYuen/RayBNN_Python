import numpy as np
import raybnn_python






x = np.array([ [-2, 4, 7, 0.1, 5], [3, 6, 9, -1, 17], [-4, 1, 12, 8, 42]   ], dtype=np.float64)
y = np.array([ [-22, 44], [73, 26]   ], dtype=np.float64)

print(x)

print(y)

print("Rust")

z = raybnn_python.magic2(x, y)

print(z)



