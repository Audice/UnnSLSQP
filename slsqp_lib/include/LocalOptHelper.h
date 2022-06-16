#pragma once
#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <cstdint>

#define NLOPT_MINF_MAX_REACHED NLOPT_STOPVAL_REACHED
#define ERR(err, opt, msg) (nlopt_set_errmsg(opt, msg) ? err : err)

typedef enum {
    NLOPT_FAILURE = -1,
    NLOPT_INVALID_ARGS = -2,
    NLOPT_OUT_OF_MEMORY = -3,
    NLOPT_ROUNDOFF_LIMITED = -4,
    NLOPT_FORCED_STOP = -5,
    NLOPT_NUM_FAILURES = -6,
    NLOPT_SUCCESS = 1,
    NLOPT_STOPVAL_REACHED = 2,
    NLOPT_FTOL_REACHED = 3,
    NLOPT_XTOL_REACHED = 4,
    NLOPT_MAXEVAL_REACHED = 5,
    NLOPT_MAXTIME_REACHED = 6,
    NLOPT_NUM_RESULTS
} nlopt_result;

/* A preconditioner, which preconditions v at x to return vpre.
(The meaning of "preconditioning" is algorithm-dependent.) */
typedef void (*nlopt_precond) (unsigned n, const double* x, const double* v, double* vpre, void* data);

typedef double (*nlopt_func) (unsigned n, const double* x,
    double* gradient, /* NULL if not needed */
    void* func_data);

typedef void (*nlopt_mfunc) (unsigned m, double* result, unsigned n, const double* x,
    double* gradient, /* NULL if not needed */
    void* func_data);

typedef void* (*nlopt_munge) (void* p);

typedef struct {
    unsigned m;             /* dimensional of constraint: mf maps R^n -> R^m */
    nlopt_func f;           /* one-dimensional constraint, requires m == 1 */
    nlopt_mfunc mf;
    nlopt_precond pre;      /* preconditioner for f (NULL if none or if mf) */
    void* f_data;
    double* tol;
} nlopt_constraint;


/* stopping criteria */
typedef struct {
    unsigned n;
    double minf_max;
    double ftol_rel;
    double ftol_abs;
    double xtol_rel;
    const double* xtol_abs;
    const double* x_weights;
    int* nevals_p, maxeval;
    double maxtime, start;
    int* force_stop;
    char** stop_msg;        /* pointer to msg string to update */
} nlopt_stopping;

typedef enum {
    /* Naming conventions:

       NLOPT_{G/L}{D/N}_*
       = global/local derivative/no-derivative optimization,
       respectively

       *_RAND algorithms involve some randomization.

       *_NOSCAL algorithms are *not* scaled to a unit hypercube
       (i.e. they are sensitive to the units of x)
     */

    NLOPT_GN_DIRECT = 0,
    NLOPT_GN_DIRECT_L,
    NLOPT_GN_DIRECT_L_RAND,
    NLOPT_GN_DIRECT_NOSCAL,
    NLOPT_GN_DIRECT_L_NOSCAL,
    NLOPT_GN_DIRECT_L_RAND_NOSCAL,

    NLOPT_GN_ORIG_DIRECT,
    NLOPT_GN_ORIG_DIRECT_L,

    NLOPT_GD_STOGO,
    NLOPT_GD_STOGO_RAND,

    NLOPT_LD_LBFGS_NOCEDAL,

    NLOPT_LD_LBFGS,

    NLOPT_LN_PRAXIS,

    NLOPT_LD_VAR1,
    NLOPT_LD_VAR2,

    NLOPT_LD_TNEWTON,
    NLOPT_LD_TNEWTON_RESTART,
    NLOPT_LD_TNEWTON_PRECOND,
    NLOPT_LD_TNEWTON_PRECOND_RESTART,

    NLOPT_GN_CRS2_LM,

    NLOPT_GN_MLSL,
    NLOPT_GD_MLSL,
    NLOPT_GN_MLSL_LDS,
    NLOPT_GD_MLSL_LDS,

    NLOPT_LD_MMA,

    NLOPT_LN_COBYLA,

    NLOPT_LN_NEWUOA,
    NLOPT_LN_NEWUOA_BOUND,

    NLOPT_LN_NELDERMEAD,
    NLOPT_LN_SBPLX,

    NLOPT_LN_AUGLAG,
    NLOPT_LD_AUGLAG,
    NLOPT_LN_AUGLAG_EQ,
    NLOPT_LD_AUGLAG_EQ,

    NLOPT_LN_BOBYQA,

    NLOPT_GN_ISRES,

    /* new variants that require local_optimizer to be set,
       not with older constants for backwards compatibility */
       NLOPT_AUGLAG,
       NLOPT_AUGLAG_EQ,
       NLOPT_G_MLSL,
       NLOPT_G_MLSL_LDS,

       NLOPT_LD_SLSQP,

       NLOPT_LD_CCSAQ,

       NLOPT_GN_ESCH,

       NLOPT_GN_AGS,

       NLOPT_NUM_ALGORITHMS        /* not an algorithm, just the number of them */
} nlopt_algorithm;




struct nlopt_opt_s;             /* opaque structure, defined internally */
typedef struct nlopt_opt_s* nlopt_opt;

typedef struct {
    char* name;
    double val;
} nlopt_opt_param;

struct nlopt_opt_s {
    nlopt_algorithm algorithm;      /* the optimization algorithm (immutable) */
    unsigned n;             /* the dimension of the problem (immutable) */

    nlopt_func f;
    void* f_data;           /* objective function to minimize */
    nlopt_precond pre;      /* optional preconditioner for f (NULL if none) */
    int maximize;           /* nonzero if we are maximizing, not minimizing */

    nlopt_opt_param* params;
    unsigned nparams;

    double* lb, * ub;        /* lower and upper bounds (length n) */

    unsigned m;             /* number of inequality constraints */
    unsigned m_alloc;       /* number of inequality constraints allocated */
    nlopt_constraint* fc;   /* inequality constraints, length m_alloc */

    unsigned p;             /* number of equality constraints */
    unsigned p_alloc;       /* number of inequality constraints allocated */
    nlopt_constraint* h;    /* equality constraints, length p_alloc */

    nlopt_munge munge_on_destroy, munge_on_copy;    /* hack for wrappers */

    /* stopping criteria */
    double stopval;         /* stop when f reaches stopval or better */
    double ftol_rel, ftol_abs;      /* relative/absolute f tolerances */
    double xtol_rel, * xtol_abs;     /* rel/abs x tolerances */
    double* x_weights;      /* weights for relative x tolerance */
    int maxeval;            /* max # evaluations */
    int numevals;           /* number of evaluations */
    double maxtime;         /* max time (seconds) */

    int force_stop;         /* if nonzero, force a halt the next time we
                               try to evaluate the objective during optimization */
                               /* when local optimization is used, we need a force_stop in the
                                  parent object to force a stop in child optimizations */
    struct nlopt_opt_s* force_stop_child;

    /* algorithm-specific parameters */
    nlopt_opt local_opt;    /* local optimizer */
    unsigned stochastic_population; /* population size for stochastic algs */
    double* dx;             /* initial step sizes (length n) for nonderivative algs */
    unsigned vector_storage;        /* max subspace dimension (0 for default) */

    void* work;             /* algorithm-specific workspace during optimization */

    char* errmsg;           /* description of most recent error */
};

static const char* nlopt_algorithm_to_string(nlopt_algorithm algorithm)
{
    switch (algorithm)
    {
    case NLOPT_GN_DIRECT: return "GN_DIRECT";
    case NLOPT_GN_DIRECT_L: return "GN_DIRECT_L";
    case NLOPT_GN_DIRECT_L_RAND: return "GN_DIRECT_L_RAND";
    case NLOPT_GN_DIRECT_NOSCAL: return "GN_DIRECT_NOSCAL";
    case NLOPT_GN_DIRECT_L_NOSCAL: return "GN_DIRECT_L_NOSCAL";
    case NLOPT_GN_DIRECT_L_RAND_NOSCAL: return "GN_DIRECT_L_RAND_NOSCAL";
    case NLOPT_GN_ORIG_DIRECT: return "GN_ORIG_DIRECT";
    case NLOPT_GN_ORIG_DIRECT_L: return "GN_ORIG_DIRECT_L";
    case NLOPT_GD_STOGO: return "GD_STOGO";
    case NLOPT_GD_STOGO_RAND: return "GD_STOGO_RAND";
    case NLOPT_LD_LBFGS_NOCEDAL: return "LD_LBFGS_NOCEDAL";
    case NLOPT_LD_LBFGS: return "LD_LBFGS";
    case NLOPT_LN_PRAXIS: return "LN_PRAXIS";
    case NLOPT_LD_VAR1: return "LD_VAR1";
    case NLOPT_LD_VAR2: return "LD_VAR2";
    case NLOPT_LD_TNEWTON: return "LD_TNEWTON";
    case NLOPT_LD_TNEWTON_RESTART: return "LD_TNEWTON_RESTART";
    case NLOPT_LD_TNEWTON_PRECOND: return "LD_TNEWTON_PRECOND";
    case NLOPT_LD_TNEWTON_PRECOND_RESTART: return "LD_TNEWTON_PRECOND_RESTART";
    case NLOPT_GN_CRS2_LM: return "GN_CRS2_LM";
    case NLOPT_GN_MLSL: return "GN_MLSL";
    case NLOPT_GD_MLSL: return "GD_MLSL";
    case NLOPT_GN_MLSL_LDS: return "GN_MLSL_LDS";
    case NLOPT_GD_MLSL_LDS: return "GD_MLSL_LDS";
    case NLOPT_LD_MMA: return "LD_MMA";
    case NLOPT_LN_COBYLA: return "LN_COBYLA";
    case NLOPT_LN_NEWUOA: return "LN_NEWUOA";
    case NLOPT_LN_NEWUOA_BOUND: return "LN_NEWUOA_BOUND";
    case NLOPT_LN_NELDERMEAD: return "LN_NELDERMEAD";
    case NLOPT_LN_SBPLX: return "LN_SBPLX";
    case NLOPT_LN_AUGLAG: return "LN_AUGLAG";
    case NLOPT_LD_AUGLAG: return "LD_AUGLAG";
    case NLOPT_LN_AUGLAG_EQ: return "LN_AUGLAG_EQ";
    case NLOPT_LD_AUGLAG_EQ: return "LD_AUGLAG_EQ";
    case NLOPT_LN_BOBYQA: return "LN_BOBYQA";
    case NLOPT_GN_ISRES: return "GN_ISRES";
    case NLOPT_AUGLAG: return "AUGLAG";
    case NLOPT_AUGLAG_EQ: return "AUGLAG_EQ";
    case NLOPT_G_MLSL: return "G_MLSL";
    case NLOPT_G_MLSL_LDS: return "G_MLSL_LDS";
    case NLOPT_LD_SLSQP: return "LD_SLSQP";
    case NLOPT_LD_CCSAQ: return "LD_CCSAQ";
    case NLOPT_GN_ESCH: return "GN_ESCH";
    case NLOPT_GN_AGS: return "GN_AGS";
    case NLOPT_NUM_ALGORITHMS: return NULL;
    }
    return NULL;
}


static const char nlopt_algorithm_names[NLOPT_NUM_ALGORITHMS][256] = {
    "DIRECT (global, no-derivative)",
    "DIRECT-L (global, no-derivative)",
    "Randomized DIRECT-L (global, no-derivative)",
    "Unscaled DIRECT (global, no-derivative)",
    "Unscaled DIRECT-L (global, no-derivative)",
    "Unscaled Randomized DIRECT-L (global, no-derivative)",
    "Original DIRECT version (global, no-derivative)",
    "Original DIRECT-L version (global, no-derivative)",
    "StoGO (global, derivative-based)",
    "StoGO with randomized search (global, derivative-based)",
    "original L-BFGS code by Nocedal et al. (NOT COMPILED)",
    "Limited-memory BFGS (L-BFGS) (local, derivative-based)",
    "Principal-axis, praxis (local, no-derivative)",
    "Limited-memory variable-metric, rank 1 (local, derivative-based)",
    "Limited-memory variable-metric, rank 2 (local, derivative-based)",
    "Truncated Newton (local, derivative-based)",
    "Truncated Newton with restarting (local, derivative-based)",
    "Preconditioned truncated Newton (local, derivative-based)",
    "Preconditioned truncated Newton with restarting (local, derivative-based)",
    "Controlled random search (CRS2) with local mutation (global, no-derivative)",
    "Multi-level single-linkage (MLSL), random (global, no-derivative)",
    "Multi-level single-linkage (MLSL), random (global, derivative)",
    "Multi-level single-linkage (MLSL), quasi-random (global, no-derivative)",
    "Multi-level single-linkage (MLSL), quasi-random (global, derivative)",
    "Method of Moving Asymptotes (MMA) (local, derivative)",
    "COBYLA (Constrained Optimization BY Linear Approximations) (local, no-derivative)",
    "NEWUOA unconstrained optimization via quadratic models (local, no-derivative)",
    "Bound-constrained optimization via NEWUOA-based quadratic models (local, no-derivative)",
    "Nelder-Mead simplex algorithm (local, no-derivative)",
    "Sbplx variant of Nelder-Mead (re-implementation of Rowan's Subplex) (local, no-derivative)",
    "Augmented Lagrangian method (local, no-derivative)",
    "Augmented Lagrangian method (local, derivative)",
    "Augmented Lagrangian method for equality constraints (local, no-derivative)",
    "Augmented Lagrangian method for equality constraints (local, derivative)",
    "BOBYQA bound-constrained optimization via quadratic models (local, no-derivative)",
    "ISRES evolutionary constrained optimization (global, no-derivative)",
    "Augmented Lagrangian method (needs sub-algorithm)",
    "Augmented Lagrangian method for equality constraints (needs sub-algorithm)",
    "Multi-level single-linkage (MLSL), random (global, needs sub-algorithm)",
    "Multi-level single-linkage (MLSL), quasi-random (global, needs sub-algorithm)",
    "Sequential Quadratic Programming (SQP) (local, derivative)",
    "CCSA (Conservative Convex Separable Approximations) with simple quadratic approximations (local, derivative)",
    "ESCH evolutionary strategy",
    "AGS (global, no-derivative)"
};


class LocalOptHelper
{
private:
    

public:
    static int nlopt_isnan(double x);
    static int nlopt_isinf(double x);
    static int nlopt_isfinite(double x);
    static int nlopt_istiny(double x);
    static double sc(double x, double smin, double smax);
    static double vector_norm(unsigned n, const double* vec, const double* w, const double* scale_min, const double* scale_max);
    static double diff_norm(unsigned n, const double* x, const double* oldx, const double* w, const double* scale_min, const double* scale_max);
    static int relstop(double vold, double vnew, double reltol, double abstol);
    static int nlopt_stop_ftol(const nlopt_stopping* s, double f, double oldf);
    static int nlopt_stop_forced(const nlopt_stopping* stop);
    static void nlopt_eval_constraint(double* result, double* grad, const nlopt_constraint* c, unsigned n, const double* x);
    static int nlopt_stop_x(const nlopt_stopping* s, const double* x, const double* oldx);
    static unsigned nlopt_count_constraints(unsigned p, const nlopt_constraint* c);
    static unsigned nlopt_max_constraint_dim(unsigned p, const nlopt_constraint* c);
    static int nlopt_stop_evals(const nlopt_stopping* s);
    static int nlopt_stop_time_(double start, double maxtime);
    static int nlopt_stop_time(const nlopt_stopping* s);
    static char* nlopt_vsprintf(char* p, const char* format, va_list ap);
    static void nlopt_stop_msg(const nlopt_stopping* s, const char* format, ...);
    static nlopt_result nlopt_set_precond_min_objective(nlopt_opt opt, nlopt_func f, nlopt_precond pre, void* f_data);
    static nlopt_result nlopt_set_min_objective(nlopt_opt opt, nlopt_func f, void* f_data);
    static nlopt_result nlopt_set_precond_max_objective(nlopt_opt opt, nlopt_func f, nlopt_precond pre, void* f_data);
    static nlopt_result nlopt_set_max_objective(nlopt_opt opt, nlopt_func f, void* f_data);
    static const char* nlopt_algorithm_name(nlopt_algorithm a);
    static nlopt_result nlopt_remove_inequality_constraints(nlopt_opt opt);
    static nlopt_result add_constraint(nlopt_opt opt,
        unsigned* m, unsigned* m_alloc, nlopt_constraint** c, unsigned fm, nlopt_func fc, nlopt_mfunc mfc, nlopt_precond pre, void* fc_data, const double* tol);
    static int inequality_ok(nlopt_algorithm algorithm);
    static const char* nlopt_set_errmsg(nlopt_opt opt, const char* format, ...);
    static nlopt_result nlopt_add_precond_inequality_constraint(nlopt_opt opt, nlopt_func fc, nlopt_precond pre, void* fc_data, double tol);
    static nlopt_result nlopt_add_inequality_constraint(nlopt_opt opt, nlopt_func fc, void* fc_data, double tol);
    static nlopt_result nlopt_add_inequality_mconstraint(nlopt_opt opt, unsigned m, nlopt_mfunc fc, void* fc_data, const double* tol);
    static int equality_ok(nlopt_algorithm algorithm);
    static nlopt_result nlopt_add_precond_equality_constraint(nlopt_opt opt, nlopt_func fc, nlopt_precond pre, void* fc_data, double tol);
    static nlopt_result nlopt_add_equality_constraint(nlopt_opt opt, nlopt_func fc, void* fc_data, double tol);
    static nlopt_result nlopt_remove_equality_constraints(nlopt_opt opt);
    static nlopt_result nlopt_add_equality_mconstraint(nlopt_opt opt, unsigned m, nlopt_mfunc fc, void* fc_data, const double* tol);


};

