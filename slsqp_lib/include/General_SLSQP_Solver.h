#pragma once
#include <vector>
#include <functional>
#include <array>
#include "slsqp.h"
#include "LocalOptHelper.h"
#include "problem_interface.h"
#include <opt.h>
#include <iostream>


typedef struct {
    double a, b;
} my_constraint_data;

//template <int N>
//constexpr inline auto ProblemFuncs = General_SLSQP_Solver::ProblemFunc<General_SLSQP_Solver::InConstr<N>>;

class General_SLSQP_Solver
{

private:

    opt optimizer;

    //Коэфициенты для расчёта градиента
    static const std::array<std::vector<int>, 4> coeff;
    static const std::array<std::vector<int>, 4> coeff2;
    static const std::array<int, 4> dd;
    //

    IProblem* curProblem;

    std::vector<double> _lowerBouds;
    std::vector<double> _upperBouds;

    int Dim;

    /// <summary>
    /// Численный расчёт градиента
    /// </summary>
    static void FiniteGradient(const std::vector<double>& x, std::vector<double>& grad, std::function<double(std::vector<double>&)> func, int accuracy);

    std::vector<std::function<double(std::vector<double>)>> ProblemFunctions;

public:
    General_SLSQP_Solver(int dim, IProblem* curProblem, std::vector<double> lowerBouds, std::vector<double> upperBouds, double xtol);


    double Solve(std::vector<double> startPoint);

    //template <int N>
    //double ProblemFunc(const std::vector<double>& x, std::vector<double>& grad, void* data)
    //{
    //    FiniteGradient(x, grad, F, 2);
    //    return F(x);
    //}

    

};

