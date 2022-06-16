#include "opt.h"




opt::opt() : o(NULL), xtmp(0), gradtmp(0), gradtmp0(0),
last_result(FAILURE), last_optf(HUGE_VAL),
forced_stop_reason(NLOPT_FORCED_STOP) {}
opt::~opt() { nlopt_destroy(o); }
opt::opt(algorithm a, unsigned n) :
    o(nlopt_create(nlopt_algorithm(a), n)),
    xtmp(0), gradtmp(0), gradtmp0(0),
    last_result(FAILURE), last_optf(HUGE_VAL),
    forced_stop_reason(NLOPT_FORCED_STOP) {
    if (!o) throw std::bad_alloc();
    nlopt_set_munge(o, free_myfunc_data, dup_myfunc_data);
}
opt::opt(const char* algo_str, unsigned n) :
    o(NULL), xtmp(0), gradtmp(0), gradtmp0(0),
    last_result(FAILURE), last_optf(HUGE_VAL),
    forced_stop_reason(NLOPT_FORCED_STOP) {
    const nlopt_algorithm a = nlopt_algorithm_from_string(algo_str);
    if (a < 0)
        throw std::invalid_argument("wrong algorithm string");
    o = nlopt_create(a, n);
    if (!o) throw std::bad_alloc();
    nlopt_set_munge(o, free_myfunc_data, dup_myfunc_data);
}


result opt::optimize(std::vector<double>& x, double& opt_f) {
    if (o && o->n != x.size())
        throw std::invalid_argument("dimension mismatch");
    forced_stop_reason = NLOPT_FORCED_STOP;
    //nlopt_result ret = nlopt_optimize(o, x.empty() ? NULL : &x[0], &opt_f);

    nlopt_stopping stop;
    stop.n = o->n;
    stop.minf_max = o->stopval;
    stop.ftol_rel = o->ftol_rel;
    stop.ftol_abs = o->ftol_abs;
    stop.xtol_rel = o->xtol_rel;
    stop.xtol_abs = o->xtol_abs;
    stop.x_weights = o->x_weights;
    o->numevals = 0;
    stop.nevals_p = &(o->numevals);
    stop.maxeval = o->maxeval;
    stop.maxtime = o->maxtime;
    stop.start = 1000000; // похоже это остановка по времени //nlopt_seconds();
    stop.force_stop = &(o->force_stop);
    stop.stop_msg = &(o->errmsg);

    SLSQP_Solver solver = SLSQP_Solver();

    nlopt_result a = solver.nlopt_slsqp(o->n, o->f, o->f_data, o->m, o->fc, o->p, o->h, o->lb, o->ub, x.empty() ? NULL : &x[0], &opt_f,
        &stop);

    last_result = result(0); //(ret);
    //last_optf = opt_f;
    //if (ret == NLOPT_FORCED_STOP)
    //    mythrow(forced_stop_reason);
    //mythrow(ret);
    return last_result;
}
// variant mainly useful for SWIG wrappers:
std::vector<double> opt::optimize(const std::vector<double>& x0) {
    std::vector<double> x(x0);
    last_result = optimize(x, last_optf);
    return x;
}
result opt::last_optimize_result() const { return last_result; }
double opt::last_optimum_value() const { return last_optf; }
// accessors:
algorithm opt::get_algorithm() const {
    if (!o) throw std::runtime_error("uninitialized nlopt::opt");
    return algorithm(o->algorithm);
}
const char* opt::get_algorithm_name() const {
    if (!o) throw std::runtime_error("uninitialized nlopt::opt");
    return LocalOptHelper::nlopt_algorithm_name(o->algorithm);
}
unsigned opt::get_dimension() const {
    if (!o) throw std::runtime_error("uninitialized nlopt::opt");
    return o->n;
}
// Set the objective function
void opt::set_min_objective(func f, void* f_data) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = f; d->f_data = f_data; d->mf = NULL; d->vf = NULL;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_set_min_objective(o, myfunc, d)); // d freed via o
}
void opt::set_min_objective(vfunc vf, void* f_data) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = NULL; d->f_data = f_data; d->mf = NULL; d->vf = vf;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_set_min_objective(o, myvfunc, d)); // d freed via o
    alloc_tmp();
}
void opt::set_max_objective(func f, void* f_data) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = f; d->f_data = f_data; d->mf = NULL; d->vf = NULL;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_set_max_objective(o, myfunc, d)); // d freed via o
}
void opt::set_max_objective(vfunc vf, void* f_data) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = NULL; d->f_data = f_data; d->mf = NULL; d->vf = vf;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_set_max_objective(o, myvfunc, d)); // d freed via o
    alloc_tmp();
}
// for internal use in SWIG wrappers -- variant that
// takes ownership of f_data, with munging for destroy/copy
void opt::set_min_objective(func f, void* f_data,
    nlopt_munge md, nlopt_munge mc) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = f; d->f_data = f_data; d->mf = NULL; d->vf = NULL;
    d->munge_destroy = md; d->munge_copy = mc;
    mythrow(LocalOptHelper::nlopt_set_min_objective(o, myfunc, d)); // d freed via o
}
void opt::set_max_objective(func f, void* f_data,
    nlopt_munge md, nlopt_munge mc) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = f; d->f_data = f_data; d->mf = NULL; d->vf = NULL;
    d->munge_destroy = md; d->munge_copy = mc;
    mythrow(LocalOptHelper::nlopt_set_max_objective(o, myfunc, d)); // d freed via o
}
// Nonlinear constraints:
void opt::remove_inequality_constraints() {
    nlopt_result ret = LocalOptHelper::nlopt_remove_inequality_constraints(o);
    mythrow(ret);
}
void opt::add_inequality_constraint(func f, void* f_data, double tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = f; d->f_data = f_data; d->mf = NULL; d->vf = NULL;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_add_inequality_constraint(o, myfunc, d, tol));
}
void opt::add_inequality_constraint(vfunc vf, void* f_data, double tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = NULL; d->f_data = f_data; d->mf = NULL; d->vf = vf;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_add_inequality_constraint(o, myvfunc, d, tol));
    alloc_tmp();
}
void opt::add_inequality_mconstraint(mfunc mf, void* f_data,
    const std::vector<double>& tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->mf = mf; d->f_data = f_data; d->f = NULL; d->vf = NULL;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_add_inequality_mconstraint(o, tol.size(), mymfunc, d,
        tol.empty() ? NULL : &tol[0]));
}
void opt::remove_equality_constraints() {
    nlopt_result ret = LocalOptHelper::nlopt_remove_equality_constraints(o);
    mythrow(ret);
}
void opt::add_equality_constraint(func f, void* f_data, double tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = f; d->f_data = f_data; d->mf = NULL; d->vf = NULL;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_add_equality_constraint(o, myfunc, d, tol));
}
void opt::add_equality_constraint(vfunc vf, void* f_data, double tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = NULL; d->f_data = f_data; d->mf = NULL; d->vf = vf;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_add_equality_constraint(o, myvfunc, d, tol));
    alloc_tmp();
}
void opt::add_equality_mconstraint(mfunc mf, void* f_data,
    const std::vector<double>& tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->mf = mf; d->f_data = f_data; d->f = NULL; d->vf = NULL;
    d->munge_destroy = d->munge_copy = NULL;
    mythrow(LocalOptHelper::nlopt_add_equality_mconstraint(o, tol.size(), mymfunc, d,
        tol.empty() ? NULL : &tol[0]));
}
// For internal use in SWIG wrappers (see also above)
void opt::add_inequality_constraint(func f, void* f_data,
    nlopt_munge md, nlopt_munge mc,
    double tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = f; d->f_data = f_data; d->mf = NULL; d->vf = NULL;
    d->munge_destroy = md; d->munge_copy = mc;
    mythrow(LocalOptHelper::nlopt_add_inequality_constraint(o, myfunc, d, tol));
}
void opt::add_equality_constraint(func f, void* f_data,
    nlopt_munge md, nlopt_munge mc,
    double tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->f = f; d->f_data = f_data; d->mf = NULL; d->vf = NULL;
    d->munge_destroy = md; d->munge_copy = mc;
    mythrow(LocalOptHelper::nlopt_add_equality_constraint(o, myfunc, d, tol));
}
void opt::add_inequality_mconstraint(mfunc mf, void* f_data,
    nlopt_munge md, nlopt_munge mc,
    const std::vector<double>& tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->mf = mf; d->f_data = f_data; d->f = NULL; d->vf = NULL;
    d->munge_destroy = md; d->munge_copy = mc;
    mythrow(LocalOptHelper::nlopt_add_inequality_mconstraint(o, tol.size(), mymfunc, d,
        tol.empty() ? NULL : &tol[0]));
}
void opt::add_equality_mconstraint(mfunc mf, void* f_data,
    nlopt_munge md, nlopt_munge mc,
    const std::vector<double>& tol) {
    myfunc_data* d = new myfunc_data;
    if (!d) throw std::bad_alloc();
    d->o = this; d->mf = mf; d->f_data = f_data; d->f = NULL; d->vf = NULL;
    d->munge_destroy = md; d->munge_copy = mc;
    mythrow(LocalOptHelper::nlopt_add_equality_mconstraint(o, tol.size(), mymfunc, d,
        tol.empty() ? NULL : &tol[0]));
}

void opt::set_lower_bounds(const std::vector<double>& lb) {
    nlopt_set_lower_bounds(o, &lb[0]);
}

nlopt_result opt::nlopt_set_lower_bounds(nlopt_opt opt, const double* lb)
{
    //nlopt_unset_errmsg(opt);
    if (opt && (opt->n == 0 || lb)) {
        unsigned int i;
        if (opt->n > 0)
            memcpy(opt->lb, lb, sizeof(double) * (opt->n));
        for (i = 0; i < opt->n; ++i)
            if (opt->lb[i] < opt->ub[i] && LocalOptHelper::nlopt_istiny(opt->ub[i] - opt->lb[i]))
                opt->lb[i] = opt->ub[i];
        return NLOPT_SUCCESS;
    }
    return NLOPT_INVALID_ARGS;
}

void opt::set_upper_bounds(const std::vector<double>& lb) {
    nlopt_set_upper_bounds(o, &lb[0]);
}

nlopt_result opt::nlopt_set_upper_bounds(nlopt_opt opt, const double* ub)
{
    //nlopt_unset_errmsg(opt);
    if (opt && (opt->n == 0 || ub)) {
        unsigned int i;
        if (opt->n > 0)
            memcpy(opt->ub, ub, sizeof(double) * (opt->n));
        for (i = 0; i < opt->n; ++i)
            if (opt->lb[i] < opt->ub[i] && LocalOptHelper::nlopt_istiny(opt->ub[i] - opt->lb[i]))
                opt->ub[i] = opt->lb[i];
        return NLOPT_SUCCESS;
    }
    return NLOPT_INVALID_ARGS;
}


void opt::set_xtol_rel(const double* xtol_rel) {
    nlopt_set_xtol_rel(o, xtol_rel);
}

nlopt_result opt::nlopt_set_xtol_rel(nlopt_opt opt, const double* xtol_rel)
{
    if (opt) {
        opt->xtol_rel = *xtol_rel;
        return NLOPT_SUCCESS;
    }
    return NLOPT_INVALID_ARGS;
}





void opt::set_xtol_abs(const double* xtol_abs) {
    nlopt_set_xtol_abs(o, xtol_abs);
}

nlopt_result opt::nlopt_set_xtol_abs(nlopt_opt opt, const double* xtol_abs)
{
    if (opt) {
        //nlopt_unset_errmsg(opt);
        if (!opt->xtol_abs && opt->n > 0) {
            opt->xtol_abs = (double*)calloc(opt->n, sizeof(double));
            if (!opt->xtol_abs) return NLOPT_OUT_OF_MEMORY;
        }
        
        memcpy(opt->xtol_abs, xtol_abs, opt->n * sizeof(double));
        return NLOPT_SUCCESS;
    }
    return NLOPT_INVALID_ARGS;
}

nlopt_opt opt::nlopt_create(nlopt_algorithm algorithm, unsigned n)
{
    nlopt_opt opt;

    if (((int)algorithm) < 0 || algorithm >= NLOPT_NUM_ALGORITHMS)
        return NULL;

    opt = (nlopt_opt)malloc(sizeof(struct nlopt_opt_s));
    if (opt) {
        opt->algorithm = algorithm;
        opt->n = n;
        opt->f = NULL;
        opt->f_data = NULL;
        opt->pre = NULL;
        opt->maximize = 0;
        opt->munge_on_destroy = opt->munge_on_copy = NULL;

        opt->lb = opt->ub = NULL;
        opt->m = opt->m_alloc = 0;
        opt->fc = NULL;
        opt->p = opt->p_alloc = 0;
        opt->h = NULL;
        opt->params = NULL;
        opt->nparams = 0;

        opt->stopval = -HUGE_VAL;
        opt->ftol_rel = opt->ftol_abs = 0;
        opt->xtol_rel = 0;
        opt->x_weights = NULL;
        opt->xtol_abs = NULL;
        opt->maxeval = 0;
        opt->numevals = 0;
        opt->maxtime = 0;
        opt->force_stop = 0;
        opt->force_stop_child = NULL;

        opt->local_opt = NULL;
        opt->stochastic_population = 0;
        opt->vector_storage = 0;
        opt->dx = NULL;
        opt->work = NULL;
        opt->errmsg = NULL;

        if (n > 0) {
            opt->lb = (double*)calloc(n, sizeof(double));
            if (!opt->lb)
                goto oom;
            opt->ub = (double*)calloc(n, sizeof(double));
            if (!opt->ub)
                goto oom;
            nlopt_set_lower_bounds1(opt, -HUGE_VAL);
            nlopt_set_upper_bounds1(opt, +HUGE_VAL);
        }
    }

    return opt;

oom:
    nlopt_destroy(opt);
    return NULL;
}

nlopt_result opt::nlopt_set_lower_bounds1(nlopt_opt opt, double lb)
{
    //nlopt_unset_errmsg(opt);
    if (opt) {
        unsigned i;
        for (i = 0; i < opt->n; ++i) {
            opt->lb[i] = lb;
            if (opt->lb[i] < opt->ub[i] && LocalOptHelper::nlopt_istiny(opt->ub[i] - opt->lb[i]))
                opt->lb[i] = opt->ub[i];
        }
        return NLOPT_SUCCESS;
    }
    return NLOPT_INVALID_ARGS;
}

nlopt_result opt::nlopt_set_upper_bounds1(nlopt_opt opt, double ub)
{
    //nlopt_unset_errmsg(opt);
    if (opt) {
        unsigned i;
        for (i = 0; i < opt->n; ++i) {
            opt->ub[i] = ub;
            if (opt->lb[i] < opt->ub[i] && LocalOptHelper::nlopt_istiny(opt->ub[i] - opt->lb[i]))
                opt->ub[i] = opt->lb[i];
        }
        return NLOPT_SUCCESS;
    }
    return NLOPT_INVALID_ARGS;
}

nlopt_algorithm opt::nlopt_algorithm_from_string(const char* name)
{
    int i;
    if (name == NULL)
        return (nlopt_algorithm)(-1);
    for (i = 0; i < NLOPT_NUM_ALGORITHMS; ++i)
    {
        if (strcmp(name, nlopt_algorithm_to_string((nlopt_algorithm)i)) == 0)
            return (nlopt_algorithm)i;
    }
    return (nlopt_algorithm)(-1);
}

void opt::nlopt_set_munge(nlopt_opt opt, nlopt_munge munge_on_destroy, nlopt_munge munge_on_copy)
{
    if (opt) {
        opt->munge_on_destroy = munge_on_destroy;
        opt->munge_on_copy = munge_on_copy;
    }
}

void opt::nlopt_destroy(nlopt_opt opt)
{
    if (opt) {
        unsigned i;
        if (opt->munge_on_destroy) {
            nlopt_munge munge = opt->munge_on_destroy;
            munge(opt->f_data);
            for (i = 0; i < opt->m; ++i)
                munge(opt->fc[i].f_data);
            for (i = 0; i < opt->p; ++i)
                munge(opt->h[i].f_data);
        }
        for (i = 0; i < opt->m; ++i)
            free(opt->fc[i].tol);
        for (i = 0; i < opt->p; ++i)
            free(opt->h[i].tol);
        for (i = 0; i < opt->nparams; ++i)
            free(opt->params[i].name);
        free(opt->params);
        free(opt->lb);
        free(opt->ub);
        free(opt->xtol_abs);
        free(opt->x_weights);
        free(opt->fc);
        free(opt->h);
        nlopt_destroy(opt->local_opt);
        free(opt->dx);
        free(opt->work);
        free(opt->errmsg);
        free(opt);
    }
}