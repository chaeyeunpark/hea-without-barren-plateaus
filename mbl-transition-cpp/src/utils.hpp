#pragma once

#include <LocalHamiltonian.hpp>
#include <ConstructSparseMat.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <span>
#include <complex>
#include <random>
#include <span>
#include <stdexcept>
#include <numbers>
#include <limits>

Eigen::VectorXcd disorderedRZDiag(std::span<const double> phi);
Eigen::VectorXd czDiag(uint32_t N, std::span<const std::array<uint32_t, 2>> wires);

/**
 * @brief
 */
inline Eigen::VectorXcd diagonalUnitary(uint32_t N, std::span<const double> phi) {
	if (phi.size() != N) {
		throw std::invalid_argument("The size of phi must be N.");
	}
	Eigen::VectorXcd res = disorderedRZDiag(phi);

	std::vector<std::array<uint32_t, 2>> cz_wires;
	for(uint32_t i = 0; i < N-1; i++) {
		cz_wires.emplace_back(std::array{i, i+1});
	}

	res.array() *= czDiag(N, cz_wires).array();
	return res;
}

Eigen::MatrixXcd rotX(uint32_t N, double theta);

template <typename Derived>
double entanglement(uint32_t N, const Eigen::MatrixBase<Derived>& vec) {
	assert(vec.size() == (1u << N));

	Eigen::Reshaped<const Derived> m = vec.reshaped(1u << (N/2), 1u << (N/2));
	Eigen::MatrixXcd rho = m * m.transpose();

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(rho, Eigen::DecompositionOptions::EigenvaluesOnly);
	Eigen::ArrayXd lambdas = solver.eigenvalues();
	lambdas = lambdas.max(std::numeric_limits<double>::min());

	return -(lambdas * log(lambdas)).sum();
}

