#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <functional>
#include <array>
#include "stronginc3_problem.h"
#include "slsqp.h"
#include "opt.h"
#include "General_SLSQP_Solver.h"
//#include "opt.h"


template <int N>
double InConstr(const std::vector<double>& x) {
    TStronginC3Problem stPr = TStronginC3Problem();
    const double* point = x.data();
    return (stPr.CalculateFunctionals(point, N));
}

void FiniteGradient(const std::vector<double>& x, std::vector<double>& grad, std::function<double(std::vector<double>&)> func, int accuracy) {
    static const std::array<std::vector<int>, 4> coeff =
    { { {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} } };
    static const std::array<std::vector<int>, 4> coeff2 =
    { { {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} } };
    static const std::array<int, 4> dd = { 2, 12, 60, 840 };

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


template <double(*F)(const std::vector<double>&)>
double myvconstraint(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    FiniteGradient(x, grad, F, 2);
    return F(x);
}

template <int N>
constexpr inline auto Func = myvconstraint<InConstr<N>>;


int main() {
    std::vector<double> lb(2);
    //lb[0] = -HUGE_VAL; lb[1] = 0;
    lb[0] = 0; lb[1] = -1.;
    std::vector<double> ub(2);
    ub[0] = 4.0; ub[1] = 3.0;

    TStronginC3Problem stPr;
    General_SLSQP_Solver slsqp_solver(2, &stPr, lb, ub, 1e-5);

    std::vector<double> x(2);
    x[0] = 1.5; x[1] = 0.5;
    //x[0] = 4; x[1] = 3;

    slsqp_solver.Solve(x);

    /*
    opt opt1("LD_SLSQP", 2);

    opt1.set_lower_bounds(lb);
    opt1.set_upper_bounds(ub);

    opt1.set_min_objective(Func<3>, nullptr);
    my_constraint_data data[2] = { {2,0}, {-1,1} };

    opt1.add_inequality_constraint(Func<0>, &data[0], 1e-6);
    opt1.add_inequality_constraint(Func<1>, &data[0], 1e-6);
    opt1.add_inequality_constraint(Func<2>, &data[1], 1e-6);
    

    opt1.set_xtol_rel(1e-5);



    std::vector<double> x(2);
    x[0] = 1.5; x[1] = 0.5;
    //x[0] = 4; x[1] = 3;
    double minf;

    try {
        opt1.optimize(x, minf);
        std::cout << "found minimum at f(" << x[0] << "," << x[1] << ") = "
            << std::setprecision(10) << minf << std::endl;
        return std::fabs(minf - 0.5443310474) < 1e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (std::exception& e) {
        std::cerr << "nlopt failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    */

    
    

}