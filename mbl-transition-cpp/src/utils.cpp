#include "utils.hpp"

Eigen::VectorXcd disorderedRZDiag(std::span<const double> phi) {
	const uint32_t N = phi.size();
	Eigen::VectorXcd res(1u << N);
	for(uint32_t s = 0; s < (1u << N); s++) {
		double elt = 0.0;
		for(uint32_t i = 0; i < N; i++) {
			if ((s >> i) & 1) {
				elt += phi[i] / 2.0;
			} else {
				elt += -phi[i] / 2.0;
			}
		}
		res(s) = std::complex<double>{std::cos(elt), std::sin(elt)};
	}
	return res;
}

Eigen::VectorXd czDiag(uint32_t N, std::span<const std::array<uint32_t, 2>> wires) {
	Eigen::VectorXd res(1u << N);
	for(uint32_t s = 0; s < (1u << N); s++) {
		double factor = 1.0;
		for(const auto [i, j]: wires) {
			if (((s >> i) & 1) && ((s >> j) & 1)) {
				factor *= -1;
			}
		}
		res(s) = factor;
	}
	return res;
}

Eigen::MatrixXcd rotX(uint32_t N, double theta) {
	constexpr std::complex<double> I{0.0, 1.0};

	Eigen::SparseMatrix<double> pauli_x(2, 2);
	pauli_x.insert(0, 1) = 1.0;
	pauli_x.insert(1, 0) = 1.0;
	pauli_x.makeCompressed();

	edp::LocalHamiltonian<double> ham(N, 2);

	for(size_t i = 0; i < N; i++) {
		ham.addOneSiteTerm(i, pauli_x);
	}

	Eigen::MatrixXd ham_mat = edp::constructMat<double>(1u << N, ham);

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	es.compute(ham_mat);
	Eigen::VectorXd evals = es.eigenvalues();
	Eigen::MatrixXd evecs = es.eigenvectors();

	Eigen::VectorXcd exp_evals = exp(-I*(theta/2.0)*evals.array().cast<std::complex<double>>());

	return evecs * exp_evals.asDiagonal() * evecs.adjoint();
}
