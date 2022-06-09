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
pub fn foo(py: Python
) -> () {
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);

    assert_eq!(vec.len(), 2);
    assert_eq!(vec[0], 1);

    assert_eq!(vec.pop(), Some(2));
    assert_eq!(vec.len(), 1);

    vec[0] = 7;
    assert_eq!(vec[0], 7);

    vec.extend([1, 2, 3].iter().copied());

    for x in &vec {
	println!("{x}");
    }
    assert_eq!(vec, [7, 1, 2, 3]);
}

#[pyfunction]
pub fn bar(py: Python
) -> () {
    let mut vec = Vec::<u64>::new();
    vec.push(1);
    vec.push(2);
    vec.push(3);
    vec.push(4);


    for i in 0..vec.len() {
	println!("vec[{}] = {}", i, vec[i]) ;

    }
}


#[pyfunction]
pub fn make_data(py: Python,
       		 z: PyReadonlyArray1<bool>,
                 x: PyReadonlyArray1<bool>,
		 phase: i64,
		 group_phase: bool
                ) -> PyResult<(PyObject, PyObject, PyObject)> {
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

	println!("1: z={:?} x={:?} num_qubits={} mut_phase={}", z_array, x_array, num_qubits, mut_phase) ;

	if group_phase {
	    let mut dotprod = 0 ;
	    for i in 0..num_qubits {
		if x_array[i] && z_array[i] {
		    dotprod += 1
		}
	    }
	    println!("2: dotprod={}", dotprod) ;
	    mut_phase += dotprod ;
	    mut_phase = mut_phase % 4 ;
	}

	println!("2: mut_phase={}", mut_phase) ;

	let dim =  1 << num_qubits ;
	let mut twos_array = Vec::<u64>::new() ;
	for i in 0..num_qubits {
	    twos_array.push(1 << i) ;
	}
	println!("3: twos_array={:?}", twos_array) ;

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

	println!("4: x_indices={} z_indices={}", x_indices, z_indices) ;

	let mut indptr = Vec::<u64>::new() ;
	for i in 0..(dim+1) {
	    indptr.push(i) ;
	}

	let mut indices = Vec::<u64>::new() ;

	for i in 0..indptr.len() {
	    indices.push(indptr[i] ^ x_indices) ;
	}
	let coeff = match phase % 4 {
	    0 => Complex64::new(1.0, 0.0),
	    1 => Complex64::new(0.0, -1.0),
	    2 => Complex64::new(-1.0, 0.0),
	    3 => Complex64::new(0.0, 1.0),
	    _ => Complex64::new(1.0, 0.0) // really should be assert!(false)
	} ;


	let mut data = Vec::new() ;
	for indp in indptr.iter() {
	    if (!indp & z_indices).count_ones() % 2 == 1 {
		data.push(-coeff) ;
	    }
	    else {
		data.push(coeff) ;
	    }
	}

	Ok((
            data.into_pyarray(py).into(),
            indices.into_pyarray(py).into(),
            indptr.into_pyarray(py).into(),
	))
    }
}

#[pymodule]
pub fn base_pauli(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(make_data))?;
    m.add_wrapped(wrap_pyfunction!(foo))?;
    m.add_wrapped(wrap_pyfunction!(bar))?;
    Ok(())
}
