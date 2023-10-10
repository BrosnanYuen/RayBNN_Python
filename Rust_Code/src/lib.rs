// We need to link `blas_src` directly, c.f. https://github.com/rust-ndarray/ndarray#how-to-enable-blas-integration
extern crate blas_src;

use numpy::ndarray::Zip;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python, PyObject, PyAny, Py};

use arrayfire;
use raybnn;

use pythonize::{depythonize, pythonize};



#[pymodule]
fn raybnn_python<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {



	#[pyfn(m)]
    fn create_start_archtecture<'py>(
        py: Python<'py>,
		input_size: u64,
		max_input_size: u64,

		output_size: u64,
		max_output_size: u64,

		max_neuron_size: u64,

		batch_size: u64,
		traj_size: u64,

		dir_path:  &str
    ) -> Py<PyAny> {

		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(

			input_size,
			max_input_size,

			output_size,
			max_output_size,

			max_neuron_size,

			batch_size,
			traj_size,

			dir_path
		);

		let obj = pythonize(py, &arch_search).unwrap();

		obj
	}















	#[pyfn(m)]
    fn magic2<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> Py<PyAny> {


		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let x_dims = x.shape().clone().to_vec();
        let x = x.to_vec().unwrap();


		let mut a = arrayfire::Array::new(&x, arrayfire::Dim4::new(&[x_dims[1] as u64, x_dims[0] as u64, 1, 1]));
		a = arrayfire::transpose(&a, false);
		arrayfire::print_gen("a".to_string(), &a, Some(6));


		let obj = pythonize(py, &a).unwrap();

		obj
	}


    #[pyfn(m)]
    fn rows_dot<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {

		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let x_dims = x.shape().clone().to_vec();
        let x = x.to_vec().unwrap();


		let mut a = arrayfire::Array::new(&x, arrayfire::Dim4::new(&[x_dims[1] as u64, x_dims[0] as u64, 1, 1]));
		a = arrayfire::transpose(&a, false);
		arrayfire::print_gen("a".to_string(), &a, Some(6));



		let mut output_vec: Vec<Vec<f64> >  = Vec::new() ;
		for i in 0..a.dims()[0]
		{
			let row = arrayfire::row(&a,i as i64);
			let mut tempvec = vec!(f64::default();row.elements());
			row.host(&mut tempvec);
			output_vec.push(tempvec);
		}


		let output = PyArray2::from_vec2(py, &output_vec).unwrap();
		output
    }
    Ok(())
}
