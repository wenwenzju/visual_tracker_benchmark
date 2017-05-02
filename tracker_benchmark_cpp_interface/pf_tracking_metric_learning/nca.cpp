#include "nca.h"

namespace nca
{
	double nca_lin_grad(Eigen::MatrixXd& x_, const Eigen::MatrixXd& X, const Eigen::VectorXi& labels, int no_dims, double lambda, Eigen::MatrixXd& dF)
	{
		//Initialize some variables
		int n = X.rows();	//n samples
		Eigen::MatrixXd x = x_;
		x.resize(x.rows()*x.cols()/no_dims, no_dims);
		double F = 0;
		dF.setZero(x.rows(), x.cols());

		//Transform the data
		Eigen::MatrixXd Y = X*x;

		//Compute conditional probabilities for current solution
		Eigen::MatrixXd sumY = Y.array().square().matrix().rowwise().sum();
		Eigen::MatrixXd sumYtmp = sumY.replicate(1, Y.rows());
		Eigen::MatrixXd P = 2*(Y*Y.adjoint())-sumYtmp.adjoint()-sumYtmp;
		P = P.array().exp();
		P.diagonal().fill(0);
		P = P.array() / P.rowwise().sum().replicate(1, P.cols()).array();
		P = P.cwiseMax(DBL_MIN);

		//Compute value of cost function and gradient
		for (int i = 0; i < n; ++i)
		{
			//Sum cost function
			Eigen::Matrix<double, -1, 1> inds = (labels.array() == labels(i)).cast<double>();
			//inds = inds.cast<double>();
			double Pi = P.row(i).cwiseProduct(inds.adjoint()).sum();
			F += Pi;

			//Sum gradient
			Eigen::MatrixXd xikA = Y.row(i).replicate(Y.rows(), 1) - Y;
			Eigen::MatrixXd xikA_ = xikA.cwiseProduct(inds.replicate(1,xikA.cols()));
			Eigen::MatrixXd xik = X.row(i).replicate(X.rows(), 1) - X;
			Eigen::MatrixXd xik_ = xik.cwiseProduct(inds.replicate(1,xik.cols()));
			Eigen::MatrixXd P_ = P.row(i).cwiseProduct(inds.adjoint());
			Eigen::MatrixXd item0 = xik.cwiseProduct(P.row(i).adjoint().replicate(1,xik.cols())).adjoint() * xikA;
			Eigen::MatrixXd item1 = xik_.cwiseProduct(P_.adjoint().replicate(1, xik_.cols())).adjoint() * xikA_;
			dF += (Pi*item0 - item1);
		}
		// Include regularization term
		if (lambda != 0) 
		{
			F = F - lambda*x.array().square().sum()/(x.rows()*x.cols());
			dF = 2*dF - 2*lambda*(x.rows()*x.cols())*x;
		}
		else dF *= 2;
		dF.resize(dF.rows()*dF.cols(),1);
		dF *= -1;

		return -F;
	}
	int minimize(Eigen::MatrixXd& X, std::vector<int>& length, const Eigen::MatrixXd& cur_X, const Eigen::VectorXi& cur_lables, int no_dims, double lambda, std::vector<double>& fX)
	{
		double INT = 0.1, EXT = 3.;
		int MAX = 20;
		double RATIO = 10., SIG = 0.1, RHO = SIG/2;

		int red, len;
		if (length.size() == 2) {red = length[1];len = length[0];}
		else {red = 1; len = length[0];}

		int i = 0; 
		bool ls_failed = false;
		Eigen::MatrixXd df0;
		double f0 = nca_lin_grad(X, cur_X, cur_lables, no_dims, lambda, df0);
		fX.clear();
		fX.push_back(f0);
		i += (len < 0);
		Eigen::MatrixXd s = -df0;
		Eigen::MatrixXd d0_ = -s.adjoint()*s;
		double d0 = d0_(0,0);
		double x3 = 1.*red/(1-d0);

		while (i < abs(len))
		{
			i += (len > 0);
			Eigen::MatrixXd X0 = X;
			double F0 = f0;
			Eigen::MatrixXd dF0 = df0;
			int M;
			if (len > 0) M = MAX;
			else M = (MAX < -len-i ? MAX : -len-i);
			double d3 = 0., f3 = 0., x2 = 0., f2 = 0., d2 = 0.;
			Eigen::MatrixXd df3;
			while (1)		//keep extrapolating as long as necessary
			{
				x2 = 0; f2 = f0; d2 = d0;
				f3 = f0;
				//Eigen::MatrixXd df2 = df0;
				df3 = df0;
				bool success = false;
				Eigen::MatrixXd tmp = X + x3*s;
				while (!success && M > 0)
				{
					M -= 1; i += (len < 0);
					
					f3 = nca_lin_grad(tmp, cur_X, cur_lables, no_dims, lambda, df3);
					success = true;
				}
				if (f3 < F0) {X0 = tmp;F0 = f3;dF0 = df3;}
				Eigen::MatrixXd d3_ = df3.adjoint()*s;
				d3 = d3_(0,0);
				if (d3 > SIG*d0 || f3 > f0 + x3*RHO*d0 || M == 0) break;

				double x1 = x2, f1 = f2, d1 = d2;
				//Eigen::MatrixXd df1 = df2;
				x2 = x3; f2 = f3; d2 = d3; //df2 = df3;
				double A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);
				double B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
				if (B*B-A*d1*(x2-x1) < 0 || B+sqrt(B*B-A*d1*(x2-x1)) == 0) x3 = x2*EXT;
				else
				{
					x3 = x1-d1*(x2-x1)*(x2-x1)/(B+sqrt(B*B-A*d1*(x2-x1)));
					if (x3 > x2*EXT || x3 < 0) x3 = x2*EXT;
					else if(x3 < x2+INT*(x2-x1)) x3 = x2+INT*(x2-x1);
				}
			}
			while ((abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0)
			{
				double x4 = 0., f4 = 0., d4 = 0.;
				if (d3 > 0 || f3 > f0+x3*RHO*d0) {x4=x3;f4=f3;d4=d3;}
				else {x2 = x3;f2=f3;d2=d3;}
				if (f4 > f0) 
				{
					if (f4-f2-d2*(x4-x2) == 0) x3 = (x2+x4)/2;
					else x3 = x2-(0.5*d2*(x4-x2)*(x4-x2))/(f4-f2-d2*(x4-x2));
				}
				else
				{
					double A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);
					double B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
					if (A == 0 || B*B-A*d2*(x4-x2)*(x4-x2) < 0) x3 = (x2+x4)/2;
					else x3 = x2+(sqrt(B*B-A*d2*(x4-x2)*(x4-x2))-B)/A;
				}
				x3 = std::max(std::min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2));
				Eigen::MatrixXd tmp = X + x3*s;
				f3 = nca_lin_grad(tmp, cur_X, cur_lables, no_dims, lambda, df3);
				if (f3 < F0) X0 = tmp;F0 = f3; dF0 = df3;
				M -= 1; i += (len < 0);
				d3 = (df3.adjoint()*s)(0,0);
			}

			if (abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0)
			{
				X += x3*s;
				f0 = f3;
				fX.push_back(f0);
				s = (df3.array().square().sum() - (df0.adjoint()*df3)(0,0))/df0.array().square().sum()*s - df3;
				df0 = df3;
				d3 = d0; d0 = (df0.adjoint()*s)(0,0);
				if (d0>0) {s=-df0;d0=-s.array().square().sum();}
				x3 = x3*std::min(RATIO, d3/(d0-DBL_MIN));
				ls_failed = false;
			}
			else
			{
				X = X0;
				f0 = F0;
				df0 = dF0;
				if (ls_failed || i > abs(len)) break;
				s = -df0;
				d0=-s.array().square().sum();
				x3 = 1./(1-d0);
				ls_failed = 1;
			}
		}

		return i;
	}
	void nca(const Eigen::MatrixXd& X, const Eigen::VectorXi& labels, int no_dims, double lambda, Eigen::MatrixXd& map)
	{
		if (0 == no_dims) no_dims = X.cols();
		
		int max_iter = 200;
		int n = X.rows(), d = X.cols();
		int batch_size = std::min(5000, n);
		int no_batches = ceil(1.*n/batch_size);
		max_iter = ceil(1.*max_iter/no_batches);

		map.setRandom(d, no_dims);
		map *= 0.01;
		bool converged = false;
		int iter = 0;
		while (iter < max_iter && !converged)
		{
			iter++;
			printf("Iteration %d of %d ...\n", iter, max_iter);
			for (int batch = 0; batch < n; batch += batch_size)
			{
				map.resize(map.rows()*map.cols(), 1);
				std::vector<double> f;
				std::vector<int> length;length.push_back(5);
				minimize(map, length, X, labels, no_dims, lambda, f);
				if (f.size() == 0 || f[f.size() - 1] - f[0] > -0.0004)
				{
					printf("Converged!\n");
					converged = true;
				}
			}
		}
	}
}