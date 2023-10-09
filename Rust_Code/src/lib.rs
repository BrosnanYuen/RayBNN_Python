// We need to link `blas_src` directly, c.f. https://github.com/rust-ndarray/ndarray#how-to-enable-blas-integration
extern crate blas_src;

use numpy::ndarray::Zip;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use arrayfire;
use raybnn;


#[pymodule]
fn raybnn_python<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn rows_dot<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {

		let x_dims = x.shape().clone().to_vec();
        let x = x.to_vec().unwrap();


		let mut a = arrayfire::Array::new(&x, arrayfire::Dim4::new(&[x_dims[1] as u64, x_dims[0] as u64, 1, 1]));
		a = arrayfire::transpose(&a, false);
		arrayfire::print_gen("a".to_string(), &a, Some(6));


		let vec2 = vec![vec![11.0, 2.0], vec![21.0, 22.0]];
		let output = PyArray2::from_vec2(py, &vec2).unwrap();
		output
    }
    Ok(())
}
