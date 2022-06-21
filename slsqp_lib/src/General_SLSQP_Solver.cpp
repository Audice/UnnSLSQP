#include "General_SLSQP_Solver.h"
#include <iomanip>

General_SLSQP_Solver::General_SLSQP_Solver(int dim, IProblem* curProblem, std::vector<double> lowerBouds, 
    std::vector<double> upperBouds, double xtol): optimizer("LD_SLSQP", dim) {
    this->curProblem = curProblem;
    this->Dim = dim;
    if (!(lowerBouds.size() == dim && upperBouds.size() == dim)) 
        throw std::invalid_argument("Invalid constraint size.");

    this->_lowerBouds = lowerBouds;
    this->_upperBouds = upperBouds;

    optimizer.set_lower_bounds(_lowerBouds);
    optimizer.set_upper_bounds(_upperBouds);

    int funcInd = 0;



    auto ProblemFunc = [](int fNumber, IProblem* curProblem) {
        return [fNumber, curProblem](const std::vector<double>& x) {
            return curProblem->CalculateFunctionals(x.data(), fNumber); 
        };
    };

    for (int i = 0; i < curProblem->GetNumberOfFunctions(); i++) {
        ProblemFunctions.push_back(ProblemFunc(i, this->curProblem));
    }


    //auto OptFunc = [](std::function<double(std::vector<double>)> pFunc) {
    //    return [pFunc](const std::vector<double>& x, std::vector<double>& grad, void* data) {
    //        FiniteGradient(x, grad, pFunc, 2);
    //        return pFunc(x);
    //    };
    //};

    auto func = [](const std::vector<double>& x, std::vector<double>& grad, void* data) {
        auto pFunc = *reinterpret_cast<std::function<double(std::vector<double>)>*>(data);
        FiniteGradient(x, grad, pFunc, 2);
        return pFunc(x);
    };

    //Установка целевой функции
    
    for (auto i = 0; i < curProblem->GetNumberOfConstraints(); ++i) {
        //auto* data = &ProblemFunctions[0];
        optimizer.add_inequality_constraint(func, &ProblemFunctions[i], 1e-6);
    }
    optimizer.set_min_objective(func, &ProblemFunctions.back());

    optimizer.set_xtol_rel(xtol);

    //Установка ограничений

}

double General_SLSQP_Solver::Solve(std::vector<double> startPoint) {
    if (startPoint.size() != this->Dim)
        throw std::invalid_argument("Invalid startPoint size.");

    std::vector<double> localStartPoint(startPoint);

    double minf;

    try {
        optimizer.optimize(localStartPoint, minf);
        std::cout << "found minimum at f(" << localStartPoint[0] << "," << localStartPoint[1] << ") = "
            << std::setprecision(10) << minf << std::endl;
        return std::fabs(minf - 0.5443310474) < 1e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (std::exception& e) {
        std::cerr << "nlopt failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

void General_SLSQP_Solver::FiniteGradient(const std::vector<double>& x, std::vector<double>& grad, std::function<double(std::vector<double>&)> func, int accuracy) {
    static const std::array<std::vector<int>, 4> coeff =
    { { {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} } };
    static const std::array<std::vector<int>, 4> coeff2 =
    { { {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} } };
    static constexpr std::array dd = { 2, 12, 60, 840 };

    // accuracy can be 0, 1, 2, 3
    const double eps = 2.2204e-6;

    grad.resize(x.size());
    std::vector<double>& xx = const_cast<std::vector<double>&>(x);

    const int innerSteps = 2 * (accuracy + 1);
    const double ddVal = dd[accuracy] * eps;

    for (size_t d = 0; d < x.size(); d++) {
        grad[d] = 0;
        for (int s = 0; s < innerSteps; ++s)
        {
            double tmp = xx[d];
            xx[d] += coeff2[accuracy][s] * eps;
            grad[d] += coeff[accuracy][s] * func(xx);
            xx[d] = tmp;
        }
        grad[d] /= ddVal;
    }
}

