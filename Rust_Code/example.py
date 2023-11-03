import numpy as np
import raybnn_python


np.random.seed(0)
x = np.random.rand(2,3,5).astype(np.float32)



print(x)

print("Rust")

z = raybnn_python.magic2(x)

print(z)



