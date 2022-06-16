/////////////////////////////////////////////////////////////////////////////
//                                                                         //
//             LOBACHEVSKY STATE UNIVERSITY OF NIZHNY NOVGOROD             //
//                                                                         //
//                       Copyright (c) 2016 by UNN.                        //
//                          All Rights Reserved.                           //
//                                                                         //
//  File:      problem_interface.h                                         //
//                                                                         //
//  Purpose:   Header file for ExaMin problem interface                    //
//                                                                         //
//                                                                         //
//  Author(s): Sovrasov V.                                                 //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////

/**
\file problem_interface.h

\authors �������� �.
\date 2016
\copyright ���� ��. �.�. ������������

\brief ���������� ������������ ������ #TIProblem

\details ���������� ������������ ������ #TIProblem � ������������� ����� ������
*/

#ifndef __PROBLEMINTERFACE_H__
#define __PROBLEMINTERFACE_H__

#include <vector>
#include <string>
#include <stdexcept>

/**
������� �����-���������, �� �������� ����������� ������, ����������� ������ �����������.

� ������ #TIProblem ������� ��������� �������, ������� ������ ���� ����������� � ����������� ������� � ��������.
*/
class IProblem
{
public:

    /// ��� ������, ������������, ���� �������� ��������� �������
    static const int OK = 0;
    /** ��� ������, ������������ �������� #GetOptimumValue � #GetOptimumPoint,
    ���� ��������������� ��������� ������ �� ����������,
    */
    static const int UNDEFINED = -1;
    /// ��� ������, ������������, ���� �������� �� ���������
    static const int ERROR = -2;

    /** ������� ���� �� ����������������� �����

    ������ ����� ����� ���������� ����� #Initialize
    \param[in] configPath ������, ���������� ���� � ����������������� ����� ������
    \return ��� ������
    */
    virtual int SetConfigPath(const std::string& configPath) = 0;

    /** ����� ����� ����������� ������

    ������ ����� ������ ���������� ����� #Initialize. ����������� ������ ���� �
    ������ ��������������.
    \param[in] dimension ����������� ������
    \return ��� ������
    */
    virtual int SetDimension(int dimension) = 0;
    ///���������� ����������� ������, ����� �������� ����� #Initialize
    virtual int GetDimension() const = 0;
    ///������������� ������
    virtual int Initialize() = 0;

    /** ����� ���������� ������� ������� ������
    */
    virtual void GetBounds(double* lower, double* upper) = 0;
    /** ����� ���������� �������� ������� ������� � ����� ����������� ��������
    \param[out] value ����������� ��������
    \return ��� ������ (#OK ��� #UNDEFINED)
    */
    virtual int GetOptimumValue(double& value) const = 0;
    /** ����� ���������� �������� ������� � ������� index � ����� ����������� ��������
    \param[out] value ����������� ��������
    \return ��� ������ (#OK ��� #UNDEFINED)
    */
    virtual int GetOptimumValue(double& value, int index) const
    {
        return IProblem::UNDEFINED;
    }
    /** ����� ���������� ���������� ����� ����������� �������� ������� �������
    \param[out] y �����, � ������� ����������� ����������� ��������
    \return ��� ������ (#OK ��� #UNDEFINED)
    */
    virtual int GetOptimumPoint(double* y) const = 0;
    /** ����� ���������� ���������� ���� ����� ����������� �������� ������� �������
    � �� ����������
    \param[out] y ���������� �����, � ������� ����������� ����������� ��������
    \param[out] n ���������� �����, � ������� ����������� ����������� ��������
    \return ��� ������ (#OK ��� #UNDEFINED)
    */
    virtual int GetAllOptimumPoint(double* y, int& n) const
    {
        return IProblem::UNDEFINED;
    }

    /** ����� ���������� ����� ����� ������� � ������ (��� ����� ����� ����������� + ����� ���������)
    \return ����� �������
    */
    virtual int GetNumberOfFunctions() const = 0;
    /** ����� ���������� ����� ����������� � ������
    \return ����� �����������
    */
    virtual int GetNumberOfConstraints() const = 0;
    /** ����� ���������� ����� ��������� � ������
    \return ����� ���������
    */
    virtual int GetNumberOfCriterions() const = 0;

    /** �����, ����������� ������� ������

    \param[in] y �����, � ������� ���������� ��������� ��������
    \param[in] fNumber ����� ����������� �������. 0 ������������� ������� �����������,
    #GetNumberOfFunctions() - 1 -- ���������� ��������
    \return �������� ������� � ��������� �������
    */
    virtual double CalculateFunctionals(const double* y, int fNumber);



    virtual bool isOptimal(const double* y, double* minv, double* maxv) { return true; }

    ///����������
    virtual ~IProblem() = 0;
};

//// ------------------------------------------------------------------------------------------------

/**
������� �����-���������, �� �������� ����������� ������, ����������� ������ �����������
������������ ��� ���������� GPU.

� ������ #IGPUProblem ������� ��������� �������, ������� ������ ���� ����������� � �����������
������� � �������� �� GPU.
*/
class IGPUProblem : public IProblem
{
public:

    /** �����, ����������� ������� ������ � ���������� ������ ������������

  \param[in] y ������, ���������� ��������������� ���������� ����������� �����, � ������� ����������
  ��������� ����������� ������
  \param[in] ����� ����������� �������
  \param[in] numPoints ���������� ������������ �����
  \param[out] values ������, � ������� ����� �������� ����������� �������� ������������
  */
    virtual void CalculateFunctionals(double* y, int fNumber, int& numPoints, double* values)
    {
        throw std::runtime_error(std::string("Required overload of the following method is not implemented: ")
            + std::string(__FUNCTION__));
    }
};


//// ------------------------------------------------------------------------------------------------

/**
������� �����-���������, �� �������� ����������� ������, ����������� ������ �����������
� ����������� �����������.

� ������ #TIProblem ������� ��������� �������, ������� ������ ���� ����������� �
����������� ������� � ������� ������ ���������� ���������� ��������.

��� ���������� ��������� �������� ��� ���������� ��������.
���������� ��������� �������� ���������� � ������� ���������� y.
*/

class IIntegerProgrammingProblem : public IGPUProblem
{
public:
    /// ��� ������, ������������, ���� ���������� �������� �������� ��� ������������� ���������
    static const int ERROR_DISCRETE_VALUE = -201;
    /// ���������� ����� ���������� ����������, ���������� ��������� ������ ��������� � ������� y
    virtual int GetNumberOfDiscreteVariable() = 0;
    /**
    ���������� ����� �������� ����������� ��������� discreteVariable.
    GetDimension() ���������� ����� ����� ����������.
    (GetDimension() - GetNumberOfDiscreteVariable()) - ����� ��������� ���������� ����������
    ��� �� ���������� ���������� == -1
    */
    virtual int GetNumberOfValues(int discreteVariable) = 0;
    /**
    ���������� �������� ����������� ��������� � ������� discreteVariable
    ���������� ��� ������.
    \param[out] values ������, � ������� ����� ��������� �������� ����������� ���������
    ������� ������� ��� ����� �������, �������� ������� ��� ������ �������.
    */
    virtual int GetAllDiscreteValues(int discreteVariable, double* values) = 0;
    /**
    ���������� �������� ����������� ��������� � ������� discreteVariable ����� ������ previousNumber
    ���������� ��� ������.
    \param[in] previousNumber - ����� �������� ����� �������� ������������ ��������
    -2 - �������� �� ���������, ���������� ��������� ��������
    -1 - ���������� ����� -1, �.�. ����� ������� �������
    \param[out] value ���������� � ������� ����������� �������� ����������� ���������
    */
    virtual int GetNextDiscreteValues(int* mCurrentDiscreteValueIndex, double& value, int discreteVariable, int previousNumber = -2) = 0;
    /// ��������� �������� �� value ���������� ��������� ��� ��������� � ������� discreteVariable
    virtual bool IsPermissibleValue(double value, int discreteVariable) = 0;
};

////
//// ------------------------------------------------------------------------------------------------
//void IGPUProblem::CalculateFunctionals(double* y, int fNumber, int& numPoints, double* values)
//{
//  throw std::runtime_error(std::string("Required overload of the following method is not implemented: ")
//    + std::string(__FUNCTION__));
//}

// ------------------------------------------------------------------------------------------------
inline double IProblem::CalculateFunctionals(const double* y, int fNumber)
{
    throw std::runtime_error(std::string("Required overload of the following method is not implemented: ")
        + std::string(__FUNCTION__));
}

// ------------------------------------------------------------------------------------------------
inline IProblem::~IProblem() {}

///��� �������-�������, ������� �������������� ������������ ����������� � �������
typedef IProblem* create_t();
///��� �������-�����������, ������� �������������� ������������ ����������� � �������
typedef void destroy_t(IProblem*);

///������� ��� �������, �������������� ������������ ����������� � �������
#ifdef WIN32
#define LIB_EXPORT_API __declspec(dllexport)
#else
#define LIB_EXPORT_API
#endif

#endif
// - end of file ----------------------------------------------------------------------------------

