// We need to link `blas_src` directly, c.f. https://github.com/rust-ndarray/ndarray#how-to-enable-blas-integration
extern crate blas_src;

use numpy::{self, IntoPyArray};
use numpy::ndarray::Zip;
use numpy::{PyReadonlyArray3, PyArray4, PyArray2, PyReadonlyArray4, PyReadonlyArray2, PyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python, PyObject, PyAny, Py};

use arrayfire;
use raybnn;

use pythonize::{depythonize, pythonize};

use nohash_hasher;

use ndarray::Axis;

#[pymodule]
fn raybnn_python<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {

	#[pyfn(m)]
    fn print_model_info<'py>(
        py: Python<'py>,
		model: Py<PyAny>
    ) {
		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();

		raybnn::neural::network_f32::print_netdata(&arch_search.neural_network.netdata);
	}



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

		directory_path:  String
    ) -> Py<PyAny> {

		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let dir_path = directory_path.clone();

		let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(

			input_size,
			max_input_size,

			output_size,
			max_output_size,

			max_neuron_size,

			batch_size,
			traj_size,

			&dir_path
		);

		let obj = pythonize(py, &arch_search).unwrap();

		obj
	}




	#[pyfn(m)]
    fn train_network<'py>(
        py: Python<'py>,

		train_x: PyReadonlyArray4<'py, f32>,
        train_y: PyReadonlyArray4<'py, f32>,

		crossval_x: PyReadonlyArray4<'py, f32>,
        crossval_y: PyReadonlyArray4<'py, f32>,

		stop_strategy_input: String,
		lr_strategy_input: String,
		lr_strategy2_input: String,

		loss_function: String,

		max_epoch: u64,
		stop_epoch: u64,
		stop_train_loss: f32,

		exit_counter_threshold: u64,
		shuffle_counter_threshold: u64,

		model: Py<PyAny>
    ) -> Py<PyAny> {

		let mut stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::NONE;

		if stop_strategy_input == "NONE"
		{
			stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::NONE;
		}
		else if stop_strategy_input == "STOP_AT_EPOCH"
		{
			stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::STOP_AT_EPOCH;
		}
		else if stop_strategy_input == "STOP_AT_TRAIN_LOSS"
		{
			stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::STOP_AT_TRAIN_LOSS;
		}
		else if stop_strategy_input == "CROSSVAL_STOPPING"
		{
			stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::CROSSVAL_STOPPING;
		}




		let mut lr_strategy = raybnn::interface::autotrain_f32::lr_strategy_type::NONE;

		if lr_strategy_input == "NONE"
		{
			lr_strategy = raybnn::interface::autotrain_f32::lr_strategy_type::NONE;
		}
		else if lr_strategy_input == "COSINE_ANNEALING"
		{
			lr_strategy = raybnn::interface::autotrain_f32::lr_strategy_type::COSINE_ANNEALING;
		}
		else if lr_strategy_input == "SHUFFLE_CONNECTIONS"
		{
			lr_strategy = raybnn::interface::autotrain_f32::lr_strategy_type::SHUFFLE_CONNECTIONS;
		}


		let mut lr_strategy2 = raybnn::interface::autotrain_f32::lr_strategy2_type::BTLS_ALPHA;

		if lr_strategy2_input == "BTLS_ALPHA"
		{
			lr_strategy2 = raybnn::interface::autotrain_f32::lr_strategy2_type::BTLS_ALPHA;
		}
		else if lr_strategy2_input == "MAX_ALPHA"
		{
			lr_strategy2 = raybnn::interface::autotrain_f32::lr_strategy2_type::MAX_ALPHA;
		}




		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();

		//Train Options
		let train_stop_options = raybnn::interface::autotrain_f32::train_network_options_type {
			stop_strategy: stop_stategy,
			lr_strategy: lr_strategy,
			lr_strategy2: lr_strategy2,

			max_epoch: max_epoch,
			stop_epoch: stop_epoch,
			stop_train_loss: stop_train_loss,

			exit_counter_threshold: exit_counter_threshold,
			shuffle_counter_threshold: shuffle_counter_threshold,
		};


		let mut alpha_max_vec = Vec::new();
		let mut loss_vec = Vec::new();
		let mut crossval_vec = Vec::new();
		let mut loss_status = raybnn::interface::autotrain_f32::loss_status_type::LOSS_OVERFLOW;

		println!("Start training");

		arrayfire::device_gc();





		let train_x_dims = train_x.shape().clone().to_vec();
		let train_y_dims = train_y.shape().clone().to_vec();

		let crossval_x_dims = crossval_x.shape().clone().to_vec();
		let crossval_y_dims = crossval_y.shape().clone().to_vec();

		let mut traindata_X: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();
		let mut traindata_Y: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();

		let mut validationdata_X: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();
		let mut validationdata_Y: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();


		let train_x = train_x.to_owned_array() ;
		let train_y = train_y.to_owned_array() ;

		let crossval_x = crossval_x.to_owned_array() ;
		let crossval_y = crossval_y.to_owned_array() ;

		for traj in 0..train_x_dims[3]
		{


			let train_x_dims = train_x.shape().clone().to_vec();
			let train_y_dims = train_y.shape().clone().to_vec();
			let X = train_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([train_x_dims[0],train_x_dims[2],train_x_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			let Y = train_y.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([train_y_dims[0],train_y_dims[2],train_y_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();




			traindata_X.insert(traj as u64, X);
			traindata_Y.insert(traj as u64, Y);
		}

		for traj in 0..crossval_x_dims[3]
		{
			let crossval_x_dims = crossval_x.shape().clone().to_vec();
			let crossval_y_dims = crossval_y.shape().clone().to_vec();
			let X = crossval_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([crossval_x_dims[0],crossval_x_dims[2],crossval_x_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			let Y = crossval_y.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([crossval_y_dims[0],crossval_y_dims[2],crossval_y_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();



			validationdata_X.insert(traj as u64, X);
			validationdata_Y.insert(traj as u64, Y);
		}




		if loss_function == "MSE"
		{
			//Train network, stop at lowest crossval
			raybnn::interface::autotrain_f32::train_network(
				&traindata_X,
				&traindata_Y,

				&validationdata_X,
				&validationdata_Y,

				raybnn::optimal::loss_f32::MSE,
				raybnn::optimal::loss_f32::MSE_grad,

				train_stop_options,

				&mut alpha_max_vec,
				&mut loss_vec,
				&mut crossval_vec,
				&mut arch_search,
				&mut loss_status
			);
		}
		else if loss_function == "softmax_cross_entropy"
		{
			raybnn::interface::autotrain_f32::train_network(
				&traindata_X,
				&traindata_Y,

				&validationdata_X,
				&validationdata_Y,

				raybnn::optimal::loss_f32::softmax_cross_entropy,
				raybnn::optimal::loss_f32::softmax_cross_entropy_grad,

				train_stop_options,

				&mut alpha_max_vec,
				&mut loss_vec,
				&mut crossval_vec,
				&mut arch_search,
				&mut loss_status
			);
		}
		else if loss_function == "sigmoid_cross_entropy"
		{
			raybnn::interface::autotrain_f32::train_network(
				&traindata_X,
				&traindata_Y,

				&validationdata_X,
				&validationdata_Y,

				raybnn::optimal::loss_f32::sigmoid_cross_entropy,
				raybnn::optimal::loss_f32::sigmoid_cross_entropy_grad,

				train_stop_options,

				&mut alpha_max_vec,
				&mut loss_vec,
				&mut crossval_vec,
				&mut arch_search,
				&mut loss_status
			);
		}


		let obj = pythonize(py, &arch_search).unwrap();

		obj
	}







	#[pyfn(m)]
    fn test_network<'py>(
        py: Python<'py>,

		test_x: PyReadonlyArray4<'py, f32>,
        test_y: PyReadonlyArray4<'py, f32>,


		model: Py<PyAny>
    )  {



		arrayfire::set_backend(arrayfire::Backend::CUDA);
		arrayfire::device_gc();

		let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();


		let test_x_dims = test_x.shape().clone().to_vec();
		let test_y_dims = test_y.shape().clone().to_vec();


		let mut validationdata_X: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();
		let mut validationdata_Y: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();


		let test_x = test_x.to_owned_array() ;
		let test_y = test_y.to_owned_array() ;

		
		for traj in 0..test_x_dims[3]
		{
			let test_x_dims = test_x.shape().clone().to_vec();
			let test_y_dims = test_y.shape().clone().to_vec();
			let X = test_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([test_x_dims[0],test_x_dims[2],test_x_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			let Y = test_y.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([test_y_dims[0],test_y_dims[2],test_y_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();



			validationdata_X.insert(traj as u64, X);
			validationdata_Y.insert(traj as u64, Y);
		}

		let mut eval_metric_out = Vec::new();
		let mut Yhat_out = nohash_hasher::IntMap::default();
		

		//Train network, stop at lowest crossval
		raybnn::interface::autotest_f32::validate_network(
			&mut validationdata_X,
			&mut validationdata_Y,

			raybnn::optimal::loss_f32::MSE, 
			&mut arch_search, 
			&mut Yhat_out, 
			&mut eval_metric_out
		);
	


	}







	#[pyfn(m)]
    fn magic2<'py>(
        py: Python<'py>,
        x: PyReadonlyArray3<'py, f32>,
    ) -> Py<PyAny> {

		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let x_dims = x.shape().clone().to_vec();
        let x = x.reshape_with_order([x_dims[0],x_dims[2],x_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();


		let mut a = arrayfire::Array::new(&x, arrayfire::Dim4::new(&[x_dims[0] as u64, x_dims[1] as u64, x_dims[2] as u64, 1]));
		arrayfire::print_gen("a".to_string(), &a, Some(6));

		let mut b = arrayfire::row(&a,1);
		b = arrayfire::col(&b,0);
		b = arrayfire::slice(&b,3);
		arrayfire::print_gen("b".to_string(), &b, Some(6));

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
