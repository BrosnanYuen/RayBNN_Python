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
        let x = x.to_vec().unwrap();
        let y = y.to_vec().unwrap();

		let a = arrayfire::Array::new(&x, arrayfire::Dim4::new(&[3, 3, 1, 1]));

		arrayfire::print_gen("a".to_string(), &a, Some(6));


		let vec2 = vec![vec![11.0, 2.0], vec![21.0, 22.0]];
		let output = PyArray2::from_vec2(py, &vec2).unwrap();
		output
    }
    Ok(())
}
