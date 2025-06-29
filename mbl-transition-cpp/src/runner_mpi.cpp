#include "utils.hpp"

#include <npy.hpp>

#include <mpi.h>

#include <complex>
#include <charconv>
#include <random>
#include <span>
#include <numbers>
#include <iostream>

class UnitaryDiagonalization {
private:
	Eigen::VectorXd evals_real_;
	Eigen::VectorXd evals_imag_;
	Eigen::MatrixXd evecs_;

public:
	explicit UnitaryDiagonalization(const Eigen::MatrixXcd& unitary) {
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
		es.compute(unitary.real());
		evecs_ = es.eigenvectors();
		evals_real_ = es.eigenvalues();
		evals_imag_ = (evecs_.adjoint() * unitary.imag() * evecs_).diagonal();
	}

	Eigen::ArrayXd spectralRatios() const {
		const uint32_t size = evals_real_.size();
		Eigen::ArrayXd res(size);

#pragma omp parallel for
		for(uint32_t i = 0; i < size; i++) {
			res(i) = std::atan2(evals_imag_(i), evals_real_(i));
		}
		return res;
	}

	const Eigen::MatrixXd& eigenvectors() const& {
		return evecs_;
	}

	Eigen::MatrixXd eigenvectors() && {
		return evecs_;
	}

	Eigen::VectorXd getNthEigenvector(uint32_t idx) {
		return evecs_.col(idx);
	}
};


int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	if (argc != 3) {
		printf("Usage: %s [N] [theta_over_pi]\n", argv[0]);
		return 1;
	}

	uint32_t N;
	double theta_over_pi;

	std::from_chars(argv[1], argv[1] + strlen(argv[1]), N);
	std::from_chars(argv[2], argv[2] + strlen(argv[2]), theta_over_pi);

	{
		std::ofstream params_out("params.json");
		params_out << "{" << "\"num_qubits\": " << N << ", " << "\"theta_over_pi\": " << theta_over_pi << "}";
		params_out.close();
	}

	const double theta = theta_over_pi * std::numbers::pi_v<double>;

	const uint32_t total_iter = 256;
	std::uniform_real_distribution<double> urd(-std::numbers::pi_v<double>, std::numbers::pi_v<double>);

	std::mt19937_64 re{1557u + mpi_rank};

	const std::vector<uint64_t> shape{1u << N};

	for(uint32_t iter_idx = mpi_rank; iter_idx < total_iter; iter_idx += mpi_size) {
		std::vector<double> phi;
		phi.reserve(N);

		for(uint32_t i = 0; i < N; i++) {
			phi.emplace_back(urd(re));
		}
		Eigen::MatrixXcd rot_x = rotX(N, theta / 2);
		Eigen::MatrixXcd unitary = rot_x * diagonalUnitary(N, phi).asDiagonal() * rot_x;
		UnitaryDiagonalization ud(unitary);

		Eigen::ArrayXd spectra = ud.spectralRatios();

		Eigen::ArrayXd ents(1u << N);

#pragma omp parallel for 
		for(uint32_t i = 0; i < (1u << N); i++) {
			ents(i) = entanglement(N, ud.getNthEigenvector(i));
		}

		std::ostringstream spectra_filename;
		spectra_filename << "spectra_" << iter_idx << ".npy";

		std::ostringstream entanglement_filename;
		entanglement_filename << "entanglement_" << iter_idx << ".npy";

		npy::SaveArrayAsNumpy(spectra_filename.str(), false, shape.size(), shape.data(), spectra.data());
		npy::SaveArrayAsNumpy(entanglement_filename.str(), false, shape.size(), shape.data(), ents.data());
	}

	MPI_Finalize();
	return 0;
}
