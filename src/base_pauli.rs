// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::time::Instant;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use pyo3::exceptions::PyException;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1, PyArray1};

/// Find the unique elements of an array.
///
/// This function is a drop-in replacement of
/// ``np.unique(array, return_index=True, return_inverse=True, axis=0)``
/// where ``array`` is a ``numpy.ndarray`` of ``dtype=u16`` and ``ndim=2``.
///
/// Note that the order of the output of this function is not sorted while ``numpy.unique``
/// returns the sorted elements.
///
/// Args:
///     array (numpy.ndarray): An array of ``dtype=u16`` and ``ndim=2``
///
/// Returns:
///     (indexes, inverses): A tuple of the following two indices.
///
///         - the indices of the input array that give the unique values
///         - the indices of the unique array that reconstruct the input array
///

#[pyfunction]
pub fn timed_make_data(py: Python,
       		       z: PyReadonlyArray1<bool>,
                       x: PyReadonlyArray1<bool>,
                       coeff: Complex64,
		       phase: i64,
		       group_phase: bool
) -> PyResult<(PyObject, PyObject, PyObject)> {
    let timings = false ;
    let now = Instant::now();
    if timings { println!("START make_data()"); }

    // Calling a slow function, it may take a while
    let rv = make_data(py, z, x, coeff, phase, group_phase) ;

    let elapsed_time = now.elapsed();
    if timings { println!("END ELAPSED make_data(): {} ms", elapsed_time.as_millis()); }
    return rv ;
}

pub fn py2rust_boolarray(v : ndarray::ArrayBase<ndarray::ViewRepr<&bool>, ndarray::Dim<[usize; 1]>>) -> Vec<bool> {
    let mut rv = Vec::<bool>::new() ;
    for i in 0..v.len() {
	rv.push(v[i]) ;
    }
    return rv ;
}

pub fn rust_make_data(z: Vec<bool>,
                      x: Vec<bool>,
                      coeff: Complex64,
		      phase: i64,
		      group_phase: bool
                ) -> std::result::Result<(Vec<Complex64>, Vec<u64>, Vec<u64>), &'static str> {
    let debug = false ;
    let timings = false ;
    let now = Instant::now();

    if z.len() != x.len() {
       Err("z and x have differing lengths")
    }
    else {
        let num_qubits = z.len() ;
	let mut mut_phase = phase ;

	if debug { println!("1: z={:?} x={:?} num_qubits={} mut_phase={}", z, x, num_qubits, mut_phase) ; }

	if group_phase {
	    let mut dotprod = 0 ;
	    for i in 0..num_qubits {
		if x[i] && z[i] {
		    dotprod += 1
		}
	    }
	    if debug { println!("2: dotprod={}", dotprod) ; }
	    mut_phase += dotprod ;
	    mut_phase = mut_phase % 4 ;
	}

	if debug { println!("2: mut_phase={}", mut_phase) ; }

	let dim =  1 << num_qubits ;
	let mut twos_array = Vec::<u64>::new() ;
	for i in 0..num_qubits {
	    twos_array.push(1 << i) ;
	}
	if debug { println!("3: twos_array={:?}", twos_array) ; }

	let mut x_indices = 0 ;
	for i in 0..num_qubits {
	    if x[i] {
		x_indices += twos_array[i] ;
	    }
	}
	let mut z_indices = 0 ;
	for i in 0..num_qubits {
	    if z[i] {
		z_indices += twos_array[i] ;
	    }
	}

	if debug { println!("4: x_indices={} z_indices={}", x_indices, z_indices) ; }


        if timings { println!("BEFORE indptr: {} ms", now.elapsed().as_millis()); }


	let mut indptr = vec![0 as u64;dim+1] ;
	for i in 0..(dim+1) {
	    indptr[i] = i as u64;
	}

        if timings { println!("BEFORE indices: {} ms", now.elapsed().as_millis()); }
	let mut indices = vec![0 as u64;indptr.len()] ;

	for i in 0..indptr.len() {
	    indices[i] = indptr[i] ^ x_indices ;
	}
	let coeff = match phase % 4 {
	    0 => Complex64::new(1.0, 0.0) * coeff,
	    1 => Complex64::new(0.0, -1.0) * coeff,
	    2 => Complex64::new(-1.0, 0.0) * coeff,
	    3 => Complex64::new(0.0, 1.0) * coeff,
	    _ => coeff // really should be assert!(false)
	} ;
	if timings { println!("coeff = {}", coeff) ; }


        if timings { println!("BEFORE data: {} ms", now.elapsed().as_millis()); }

	let mut data = Vec::new() ;
	for indp in indptr.iter() {
	    if debug { println!("indp[] = {}", indp) ; }
	    if (indp & z_indices).count_ones() % 2 == 1 {
		data.push(-coeff) ;
	    }
	    else {
		data.push(coeff) ;
	    }
	}
        if timings { println!("AFTER data: {} ms", now.elapsed().as_millis()); }

	Ok((
            data,
            indices,
            indptr,
	))
    }
}

#[pyfunction]
pub fn timed_old_make_data(py: Python,
       		           z: PyReadonlyArray1<bool>,
                           x: PyReadonlyArray1<bool>,
                           coeff: Complex64,
		           phase: i64,
		           group_phase: bool
) -> PyResult<(PyObject, PyObject, PyObject)> {
    let timings = false ;
    let now = Instant::now();
    if timings { println!("START old_make_data()"); }

    // Calling a slow function, it may take a while
    let rv = old_make_data(py, z, x, coeff, phase, group_phase) ;

    if timings { println!("END ELAPSED old_make_data(): {} ms", now.elapsed().as_millis()); }
    return rv ;
}

#[pyfunction]
pub fn old_make_data(py: Python,
       		     z: PyReadonlyArray1<bool>,
                     x: PyReadonlyArray1<bool>,
                     coeff: Complex64,
		     phase: i64,
		     group_phase: bool
                ) -> PyResult<(PyObject, PyObject, PyObject)> {
    let debug = false ;
    let timings = false ;
    let now = Instant::now();

    let z_array = z.as_array() ;
    let x_array = x.as_array() ;
    let x_shape = x_array.shape() ;
    let z_shape = z_array.shape() ;

    if z_shape[0] != x_shape[0] {
       Err(PyException::new_err("z and x have differing lengths"))
    }
    else {
        let num_qubits = z_shape[0] ;
	let mut mut_phase = phase ;

	if debug { println!("1: z={:?} x={:?} num_qubits={} mut_phase={}", z_array, x_array, num_qubits, mut_phase) ; }

	if group_phase {
	    let mut dotprod = 0 ;
	    for i in 0..num_qubits {
		if x_array[i] && z_array[i] {
		    dotprod += 1
		}
	    }
	    if debug { println!("2: dotprod={}", dotprod) ; }
	    mut_phase += dotprod ;
	    mut_phase = mut_phase % 4 ;
	}

	if debug { println!("2: mut_phase={}", mut_phase) ; }

	let dim =  1 << num_qubits ;
	let mut twos_array = Vec::<u64>::new() ;
	for i in 0..num_qubits {
	    twos_array.push(1 << i) ;
	}
	if debug { println!("3: twos_array={:?}", twos_array) ; }

	let mut x_indices = 0 ;
	for i in 0..num_qubits {
	    if x_array[i] {
		x_indices += twos_array[i] ;
	    }
	}
	let mut z_indices = 0 ;
	for i in 0..num_qubits {
	    if z_array[i] {
		z_indices += twos_array[i] ;
	    }
	}

	if debug { println!("4: x_indices={} z_indices={}", x_indices, z_indices) ; }


        if timings { println!("BEFORE indptr: {} ms", now.elapsed().as_millis()); }


	let mut indptr = vec![0 as u64;dim+1] ;
	for i in 0..(dim+1) {
	    indptr[i] = i as u64;
	}

        if timings { println!("BEFORE indices: {} ms", now.elapsed().as_millis()); }
	let mut indices = vec![0 as u64;indptr.len()] ;

	for i in 0..indptr.len() {
	    indices[i] = indptr[i] ^ x_indices ;
	}
	let coeff = match phase % 4 {
	    0 => Complex64::new(1.0, 0.0) * coeff,
	    1 => Complex64::new(0.0, -1.0) * coeff,
	    2 => Complex64::new(-1.0, 0.0) * coeff,
	    3 => Complex64::new(0.0, 1.0) * coeff,
	    _ => coeff // really should be assert!(false)
	} ;
	if debug { println!("coeff = {}", coeff) ; }


        if timings { println!("BEFORE data: {} ms", now.elapsed().as_millis()); }

	let mut data = Vec::new() ;
	for indp in indptr.iter() {
	    if debug { println!("indp[] = {}", indp) ; }
	    if (indp & z_indices).count_ones() % 2 == 1 {
		data.push(-coeff) ;
	    }
	    else {
		data.push(coeff) ;
	    }
	}
        if timings { println!("AFTER data: {} ms", now.elapsed().as_millis()); }

	Ok((
            data.into_pyarray(py).into(),
            indices.into_pyarray(py).into(),
            indptr.into_pyarray(py).into(),
	))
    }
}


#[pyfunction]
pub fn make_data(py: Python,
       		 z: PyReadonlyArray1<bool>,
                 x: PyReadonlyArray1<bool>,
                 coeff: Complex64,
		 phase: i64,
		 group_phase: bool
                ) -> PyResult<(PyObject, PyObject, PyObject)> {

    let rv = rust_make_data(py2rust_boolarray(z.as_array()), py2rust_boolarray(x.as_array()), coeff, phase, group_phase) ;
    match rv {
	std::result::Result::Err(msg) => Err(PyException::new_err(msg)),
	std::result::Result::Ok((data, indices, indptr)) =>
	    Ok((
		data.into_pyarray(py).into(),
		indices.into_pyarray(py).into(),
		indptr.into_pyarray(py).into(),
	    ))
    }
}

use qrusty::Pauli;
use qrusty::PauliList;
use qrusty::SparsePauliOp;

#[pyfunction]
pub fn qrusty_Pauli_make_data(
    py: Python,
    s : String,
    coeff: Complex64,
) -> PyResult<(u64, PyObject, PyObject, PyObject)> {

    let p = Pauli::new(&s).unwrap() ;
    let mut sp_mat = p.to_matrix() ;
    sp_mat.scale(coeff) ;
    let num_qubits = p.num_qubits() as u64 ;
    let (indptr, indices, data) = sp_mat.into_raw_storage();

    Ok((
        num_qubits,
	data.into_pyarray(py).into(),
	indices.into_pyarray(py).into(),
	indptr.into_pyarray(py).into(),
    ))
}

#[pyfunction]
pub fn qrusty_SparsePauliOp_make_data(
    py: Python,
    labels : Vec<String>,
    coeffs: Vec<Complex64>,
) -> PyResult<(u64, PyObject, PyObject, PyObject)> {

    if labels.len() != coeffs.len() {
       Err(PyException::new_err("labels and coeffs have differing lengths"))
    }
    else {
        let mut l : Vec<&str> = Vec::new() ;
        for s in labels.iter() { l.push(s) } ;
        let spop = SparsePauliOp::new(
            PauliList::from_labels_str(&l).unwrap(),
            coeffs) ;

        let spop = spop.unwrap() ;
        let sp_mat = spop.to_matrix() ;

        let num_qubits = spop.num_qubits() as u64 ;
        let (indptr, indices, data) = sp_mat.into_raw_storage();

    Ok((
        num_qubits,
	data.into_pyarray(py).into(),
	indices.into_pyarray(py).into(),
	indptr.into_pyarray(py).into(),
    ))
    }
}


#[pyfunction]
pub fn qrusty_SparsePauliOp_make_data_binary(
    py: Python,
    labels : Vec<String>,
    coeffs: Vec<Complex64>,
) -> PyResult<(u64, PyObject, PyObject, PyObject)> {

    if labels.len() != coeffs.len() {
       Err(PyException::new_err("labels and coeffs have differing lengths"))
    }
    else {
        let mut l : Vec<&str> = Vec::new() ;
        for s in labels.iter() { l.push(s) } ;
        let spop = SparsePauliOp::new(
            PauliList::from_labels_str(&l).unwrap(),
            coeffs) ;

        let spop = spop.unwrap() ;
        let sp_mat = spop.to_matrix_binary() ;

        let num_qubits = spop.num_qubits() as u64 ;
        let (indptr, indices, data) = sp_mat.into_raw_storage();

    Ok((
        num_qubits,
	data.into_pyarray(py).into(),
	indices.into_pyarray(py).into(),
	indptr.into_pyarray(py).into(),
    ))
    }
}


#[pyfunction]
pub fn qrusty_SparsePauliOp_make_data_accel(
    py: Python,
    labels : Vec<String>,
    coeffs: Vec<Complex64>,
) -> PyResult<(u64, PyObject, PyObject, PyObject)> {

    if labels.len() != coeffs.len() {
       Err(PyException::new_err("labels and coeffs have differing lengths"))
    }
    else {
        let mut l : Vec<&str> = Vec::new() ;
        for s in labels.iter() { l.push(s) } ;
        let spop = SparsePauliOp::new(
            PauliList::from_labels_str(&l).unwrap(),
            coeffs) ;

        let spop = spop.unwrap() ;
        let sp_mat = spop.to_matrix_accel() ;

        let num_qubits = spop.num_qubits() as u64 ;
        let (indptr, indices, data) = sp_mat.into_raw_storage();

    Ok((
        num_qubits,
	data.into_pyarray(py).into(),
	indices.into_pyarray(py).into(),
	indptr.into_pyarray(py).into(),
    ))
    }
}


#[pyfunction]
pub fn qrusty_SparsePauliOp_make_data_rayon(
    py: Python,
    labels : Vec<String>,
    coeffs: Vec<Complex64>,
) -> PyResult<(u64, PyObject, PyObject, PyObject)> {

    if labels.len() != coeffs.len() {
       Err(PyException::new_err("labels and coeffs have differing lengths"))
    }
    else {
        let mut l : Vec<&str> = Vec::new() ;
        for s in labels.iter() { l.push(s) } ;
        let spop = SparsePauliOp::new(
            PauliList::from_labels_str(&l).unwrap(),
            coeffs) ;

        let spop = spop.unwrap() ;
        let sp_mat = spop.to_matrix_rayon() ;

        let num_qubits = spop.num_qubits() as u64 ;
        let (indptr, indices, data) = sp_mat.into_raw_storage();

    Ok((
        num_qubits,
	data.into_pyarray(py).into(),
	indices.into_pyarray(py).into(),
	indptr.into_pyarray(py).into(),
    ))
    }
}


#[pymodule]
pub fn base_pauli(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(make_data))?;
    m.add_wrapped(wrap_pyfunction!(timed_make_data))?;
    m.add_wrapped(wrap_pyfunction!(old_make_data))?;
    m.add_wrapped(wrap_pyfunction!(timed_old_make_data))?;
    m.add_wrapped(wrap_pyfunction!(qrusty_Pauli_make_data))?;
    m.add_wrapped(wrap_pyfunction!(qrusty_SparsePauliOp_make_data))?;
    m.add_wrapped(wrap_pyfunction!(qrusty_SparsePauliOp_make_data_binary))?;
    m.add_wrapped(wrap_pyfunction!(qrusty_SparsePauliOp_make_data_accel))?;
    m.add_wrapped(wrap_pyfunction!(qrusty_SparsePauliOp_make_data_rayon))?;
    Ok(())
}
