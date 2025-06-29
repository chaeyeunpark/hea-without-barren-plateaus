#include "utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<std::complex<double>> createUnitary(py::array_t<double> phi, double theta) {
	py::buffer_info phi_buffer = phi.request();
	if (phi_buffer.ndim != 1) {
		throw std::invalid_argument("phi must be a one-dimensional array");
	}
	const uint32_t N = phi_buffer.size;

	Eigen::MatrixXcd rot_x = rotX(N, theta / 2);

	const uint32_t mat_dim = 1u << N;

	auto res = py::array_t<std::complex<double>>({mat_dim, mat_dim});
	py::buffer_info res_buffer = res.request();

	Eigen::Map<Eigen::MatrixXcd> unitary(static_cast<std::complex<double>*>(res_buffer.ptr), mat_dim, mat_dim);
	unitary = rot_x * diagonalUnitary(N, {static_cast<const double*>(phi_buffer.ptr), N}).asDiagonal() * rot_x;

	return res;
}

py::array_t<std::complex<double>> ibmUnitary(py::array_t<double> phi, double theta) {
	py::buffer_info phi_buffer = phi.request();

	const uint32_t N = phi_buffer.size;

	std::vector<std::array<uint32_t, 2>> even_wires;
	std::vector<std::array<uint32_t, 2>> odd_wires;

	for(uint32_t i = 0; i < N; i += 2) {
		even_wires.emplace_back(std::array{i, i+1});
	}

	for(uint32_t i = 1; i + 2 < N; i += 2) {
		odd_wires.emplace_back(std::array{i, i+1});
	}

	Eigen::VectorXd cz_even_diag = czDiag(N, even_wires);
	Eigen::VectorXd cz_odd_diag = czDiag(N, odd_wires);

	Eigen::MatrixXcd rot_x = rotX(N, theta);
	Eigen::VectorXcd disordered_rz = disorderedRZDiag({static_cast<const double*>(phi_buffer.ptr), N});

	const uint32_t mat_dim = 1u << N;
	auto res = py::array_t<std::complex<double>>({mat_dim, mat_dim});
	py::buffer_info res_buffer = res.request();

	Eigen::Map<Eigen::MatrixXcd> unitary(static_cast<std::complex<double>*>(res_buffer.ptr), mat_dim, mat_dim);

	unitary = cz_even_diag.asDiagonal();
	unitary *= rot_x;
	unitary *= disordered_rz.asDiagonal();
	unitary *= cz_odd_diag.asDiagonal();
	unitary *= rot_x;
	unitary *= disordered_rz.asDiagonal();
	return res;
}


PYBIND11_MODULE(kicked_ising_mbl, m) {
	m.def("create_unitary", createUnitary)
	.def("ibm_unitary", ibmUnitary);
}
