#ifndef __NCA_H__
#define __NCA_H__

#include "Eigen/Core"
#include <vector>

namespace nca
{
	double nca_lin_grad(Eigen::MatrixXd& x_, const Eigen::MatrixXd& X, const Eigen::VectorXi& labels, int no_dims, double lambda, Eigen::MatrixXd& dF);
	int minimize(Eigen::MatrixXd& X, std::vector<int>& length, const Eigen::MatrixXd& cur_X, const Eigen::VectorXi& cur_lables, int no_dims, double lambda, std::vector<double>& fX);
	void nca(const Eigen::MatrixXd& X, const Eigen::VectorXi& labels, int no_dims, double lambda, Eigen::MatrixXd& map);
}

#endif