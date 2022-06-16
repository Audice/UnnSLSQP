#include "LocalOptHelper.h"



int LocalOptHelper::nlopt_isnan(double x)
{
#if defined(HAVE_ISNAN)
    return isnan(x);
#elif defined(_WIN32)
    return _isnan(x);
#else
    return (x != x);            /* might fail with aggressive optimization */
#endif
}

int LocalOptHelper::nlopt_isinf(double x)
{
    return (fabs(x) >= HUGE_VAL * 0.99)
#if defined(HAVE_ISINF)
        || isinf(x)
#else
        || (!nlopt_isnan(x) && nlopt_isnan(x - x))
#endif
        ;
}

int LocalOptHelper::nlopt_isfinite(double x)
{
    return (fabs(x) <= DBL_MAX)
#if defined(HAVE_ISFINITE)
        || isfinite(x)
#elif defined(_WIN32)
        || _finite(x)
#endif
        ;
}

int LocalOptHelper::nlopt_istiny(double x)
{
    if (x == 0.0)
        return 1;
    else {
#if defined(HAVE_FPCLASSIFY)
        return fpclassify(x) == FP_SUBNORMAL;
#elif defined(_WIN32)
        int c = _fpclass(x);
        return c == _FPCLASS_ND || c == _FPCLASS_PD;
#else
        return fabs(x) < 2.2250738585072014e-308;       /* assume IEEE 754 double */
#endif
    }
}




double LocalOptHelper::sc(double x, double smin, double smax)
{
    return smin + x * (smax - smin);
}

double LocalOptHelper::vector_norm(unsigned n, const double* vec, const double* w, const double* scale_min, const double* scale_max)
{
    unsigned i;
    double ret = 0;
    if (scale_min && scale_max) {
        if (w)
            for (i = 0; i < n; i++)
                ret += w[i] * fabs(sc(vec[i], scale_min[i], scale_max[i]));
        else
            for (i = 0; i < n; i++)
                ret += fabs(sc(vec[i], scale_min[i], scale_max[i]));
    }
    else {
        if (w)
            for (i = 0; i < n; i++)
                ret += w[i] * fabs(vec[i]);
        else
            for (i = 0; i < n; i++)
                ret += fabs(vec[i]);
    }
    return ret;
}



double LocalOptHelper::diff_norm(unsigned n, const double* x, const double* oldx, const double* w, const double* scale_min, const double* scale_max)
{
    unsigned i;
    double ret = 0;
    if (scale_min && scale_max) {
        if (w)
            for (i = 0; i < n; i++)
                ret += w[i] * fabs(sc(x[i], scale_min[i], scale_max[i]) - sc(oldx[i], scale_min[i], scale_max[i]));
        else
            for (i = 0; i < n; i++)
                ret += fabs(sc(x[i], scale_min[i], scale_max[i]) - sc(oldx[i], scale_min[i], scale_max[i]));
    }
    else {
        if (w)
            for (i = 0; i < n; i++)
                ret += w[i] * fabs(x[i] - oldx[i]);
        else
            for (i = 0; i < n; i++)
                ret += fabs(x[i] - oldx[i]);
    }
    return ret;
}


int LocalOptHelper::relstop(double vold, double vnew, double reltol, double abstol)
{
    if (nlopt_isinf(vold))
        return 0;
    return (fabs(vnew - vold) < abstol || fabs(vnew - vold) < reltol * (fabs(vnew) + fabs(vold)) * 0.5 || (reltol > 0 && vnew == vold));        /* catch vnew == vold == 0 */
}

int LocalOptHelper::nlopt_stop_ftol(const nlopt_stopping* s, double f, double oldf)
{
    return (relstop(oldf, f, s->ftol_rel, s->ftol_abs));
}

int LocalOptHelper::nlopt_stop_forced(const nlopt_stopping* stop)
{
    return stop->force_stop && *(stop->force_stop);
}

void LocalOptHelper::nlopt_eval_constraint(double* result, double* grad, const nlopt_constraint* c, unsigned n, const double* x)
{
    if (c->f)
        result[0] = c->f(n, x, grad, c->f_data);
    else
        c->mf(c->m, result, n, x, grad, c->f_data);
}

int LocalOptHelper::nlopt_stop_x(const nlopt_stopping* s, const double* x, const double* oldx)
{
    unsigned i;
    if (diff_norm(s->n, x, oldx, s->x_weights, NULL, NULL) < s->xtol_rel * vector_norm(s->n, x, s->x_weights, NULL, NULL))
        return 1;
    if (!s->xtol_abs) return 0;
    for (i = 0; i < s->n; ++i)
        if (fabs(x[i] - oldx[i]) >= s->xtol_abs[i])
            return 0;
    return 1;
}

unsigned LocalOptHelper::nlopt_count_constraints(unsigned p, const nlopt_constraint* c)
{
    unsigned i, count = 0;
    for (i = 0; i < p; ++i)
        count += c[i].m;
    return count;
}

unsigned LocalOptHelper::nlopt_max_constraint_dim(unsigned p, const nlopt_constraint* c)
{
    unsigned i, max_dim = 0;
    for (i = 0; i < p; ++i)
        if (c[i].m > max_dim)
            max_dim = c[i].m;
    return max_dim;
}

int LocalOptHelper::nlopt_stop_evals(const nlopt_stopping* s)
{
    return (s->maxeval > 0 && *(s->nevals_p) >= s->maxeval);
}

int LocalOptHelper::nlopt_stop_time_(double start, double maxtime)
{
    //return (maxtime > 0 && nlopt_seconds() - start >= maxtime);
    return (maxtime > 0);
}


/* return time in seconds since some arbitrary point in the past */
/*
double nlopt_seconds(void)
{
    static THREADLOCAL int start_inited = 0;  
#if defined(HAVE_GETTIMEOFDAY)
    static THREADLOCAL struct timeval start;
    struct timeval tv;
    if (!start_inited) {
        start_inited = 1;
        gettimeofday(&start, NULL);
    }
    gettimeofday(&tv, NULL);
    return (tv.tv_sec - start.tv_sec) + 1.e-6 * (tv.tv_usec - start.tv_usec);
#elif defined(HAVE_TIME)
    return time(NULL);
#elif defined(_WIN32) || defined(__WIN32__)
    static THREADLOCAL ULONGLONG start;
    FILETIME ft;
    if (!start_inited) {
        start_inited = 1;
        GetSystemTimeAsFileTime(&ft);
        start = (((ULONGLONG)ft.dwHighDateTime) << 32) + ft.dwLowDateTime;
    }
    GetSystemTimeAsFileTime(&ft);
    return 100e-9 * (((((ULONGLONG)ft.dwHighDateTime) << 32) + ft.dwLowDateTime) - start);
#else
    static THREADLOCAL clock_t start;
    if (!start_inited) {
        start_inited = 1;
        start = clock();
    }
    return (clock() - start) * 1.0 / CLOCKS_PER_SEC;
#endif
}

unsigned long nlopt_time_seed(void)
{
#if defined(HAVE_GETTIMEOFDAY)
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec ^ tv.tv_usec);
#elif defined(HAVE_TIME)
    return time(NULL);
#elif defined(_WIN32) || defined(__WIN32__)
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    return ft.dwHighDateTime ^ ft.dwLowDateTime;
#else
    return clock();
#endif
}
*/

int LocalOptHelper::nlopt_stop_time(const nlopt_stopping* s)
{
    return nlopt_stop_time_(s->start, s->maxtime);
}


char* LocalOptHelper::nlopt_vsprintf(char* p, const char* format, va_list ap)
{
    unsigned long long len = strlen(format) + 128;
    unsigned long long ret;

    p = (char*)realloc(p, len);
    if (!p)
        abort();

    /* TODO: check HAVE_VSNPRINTF, and fallback to vsprintf otherwise */
    while ((ret = vsnprintf(p, len, format, ap)) < 0 || ret >= len) {
        /* C99 vsnprintf returns the required number of bytes (excluding \0)
           if the buffer is too small; older versions (e.g. MS) return -1 */
        len = ret >= 0 ? (size_t)(ret + 1) : (len * 3) >> 1;
        p = (char*)realloc(p, len);
        if (!p)
            abort();
    }
    return p;
}

void LocalOptHelper::nlopt_stop_msg(const nlopt_stopping* s, const char* format, ...)
{
    va_list ap;
    if (s->stop_msg) {
        va_start(ap, format);
        *(s->stop_msg) = nlopt_vsprintf(*(s->stop_msg), format, ap);
        va_end(ap);
    }
}

nlopt_result LocalOptHelper::nlopt_set_precond_min_objective(nlopt_opt opt, nlopt_func f, nlopt_precond pre, void* f_data)
{
    if (opt) {
        //nlopt_unset_errmsg(opt);
        if (opt->munge_on_destroy)
            opt->munge_on_destroy(opt->f_data);
        opt->f = f;
        opt->f_data = f_data;
        opt->pre = pre;
        opt->maximize = 0;
        if (nlopt_isinf(opt->stopval) && opt->stopval > 0)
            opt->stopval = -HUGE_VAL;   /* switch default from max to min */
        return NLOPT_SUCCESS;
    }
    return NLOPT_INVALID_ARGS;
}

nlopt_result LocalOptHelper::nlopt_set_min_objective(nlopt_opt opt, nlopt_func f, void* f_data)
{
    return nlopt_set_precond_min_objective(opt, f, NULL, f_data);
}

nlopt_result LocalOptHelper::nlopt_set_precond_max_objective(nlopt_opt opt, nlopt_func f, nlopt_precond pre, void* f_data)
{
    if (opt) {
        //nlopt_unset_errmsg(opt);
        if (opt->munge_on_destroy)
            opt->munge_on_destroy(opt->f_data);
        opt->f = f;
        opt->f_data = f_data;
        opt->pre = pre;
        opt->maximize = 1;
        if (nlopt_isinf(opt->stopval) && opt->stopval < 0)
            opt->stopval = +HUGE_VAL;   /* switch default from min to max */
        return NLOPT_SUCCESS;
    }
    return NLOPT_INVALID_ARGS;
}

nlopt_result LocalOptHelper::nlopt_set_max_objective(nlopt_opt opt, nlopt_func f, void* f_data)
{
    return nlopt_set_precond_max_objective(opt, f, NULL, f_data);
}

const char* LocalOptHelper::nlopt_algorithm_name(nlopt_algorithm a)
{
    if (((int)a) < 0 || a >= NLOPT_NUM_ALGORITHMS)
        return "UNKNOWN";
    return nlopt_algorithm_names[a];
}


nlopt_result LocalOptHelper::nlopt_remove_inequality_constraints(nlopt_opt opt)
{
    unsigned i;
    //nlopt_unset_errmsg(opt);
    if (!opt)
        return NLOPT_INVALID_ARGS;
    if (opt->munge_on_destroy) {
        nlopt_munge munge = opt->munge_on_destroy;
        for (i = 0; i < opt->m; ++i)
            munge(opt->fc[i].f_data);
    }
    for (i = 0; i < opt->m; ++i)
        free(opt->fc[i].tol);
    free(opt->fc);
    opt->fc = NULL;
    opt->m = opt->m_alloc = 0;
    return NLOPT_SUCCESS;
}

nlopt_result LocalOptHelper::add_constraint(nlopt_opt opt,
    unsigned* m, unsigned* m_alloc, nlopt_constraint** c, unsigned fm, nlopt_func fc, nlopt_mfunc mfc, nlopt_precond pre, void* fc_data, const double* tol)
{
    double* tolcopy;
    unsigned i;

    if ((fc && mfc) || (fc && fm != 1) || (!fc && !mfc))
        return NLOPT_INVALID_ARGS;
    if (tol)
        for (i = 0; i < fm; ++i)
            if (tol[i] < 0)
                return NLOPT_INVALID_ARGS; // ERR(NLOPT_INVALID_ARGS, opt, "negative constraint tolerance");

    tolcopy = (double*)malloc(sizeof(double) * fm);
    if (fm && !tolcopy)
        return NLOPT_OUT_OF_MEMORY;
    if (tol)
        memcpy(tolcopy, tol, sizeof(double) * fm);
    else
        for (i = 0; i < fm; ++i)
            tolcopy[i] = 0;

    *m += 1;
    if (*m > *m_alloc) {
        /* allocate by repeated doubling so that
           we end up with O(log m) mallocs rather than O(m). */
        *m_alloc = 2 * (*m);
        *c = (nlopt_constraint*)realloc(*c, sizeof(nlopt_constraint)
            * (*m_alloc));
        if (!*c) {
            *m_alloc = *m = 0;
            free(tolcopy);
            return NLOPT_OUT_OF_MEMORY;
        }
    }

    (*c)[*m - 1].m = fm;
    (*c)[*m - 1].f = fc;
    (*c)[*m - 1].pre = pre;
    (*c)[*m - 1].mf = mfc;
    (*c)[*m - 1].f_data = fc_data;
    (*c)[*m - 1].tol = tolcopy;
    return NLOPT_SUCCESS;
}

int LocalOptHelper::inequality_ok(nlopt_algorithm algorithm)
{
    /* nonlinear constraints are only supported with some algorithms */
    return (algorithm == NLOPT_LD_MMA || algorithm == NLOPT_LD_CCSAQ || algorithm == NLOPT_LD_SLSQP || algorithm == NLOPT_LN_COBYLA
        || algorithm == NLOPT_GN_ISRES || algorithm == NLOPT_GN_ORIG_DIRECT || algorithm == NLOPT_GN_ORIG_DIRECT_L || algorithm == NLOPT_GN_AGS);
}

const char* LocalOptHelper::nlopt_set_errmsg(nlopt_opt opt, const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    opt->errmsg = nlopt_vsprintf(opt->errmsg, format, ap);
    va_end(ap);
    return opt->errmsg;
}



nlopt_result LocalOptHelper::nlopt_add_precond_inequality_constraint(nlopt_opt opt, nlopt_func fc, nlopt_precond pre, void* fc_data, double tol)
{
    nlopt_result ret;
    //nlopt_unset_errmsg(opt);
    if (!opt)
        ret = NLOPT_INVALID_ARGS;
    else if (!inequality_ok(opt->algorithm))
        ret = ERR(NLOPT_INVALID_ARGS, opt, "invalid algorithm for constraints");
    else
        ret = add_constraint(opt, &opt->m, &opt->m_alloc, &opt->fc, 1, fc, NULL, pre, fc_data, &tol);
    if (ret < 0 && opt && opt->munge_on_destroy)
        opt->munge_on_destroy(fc_data);
    return ret;
}

nlopt_result LocalOptHelper::nlopt_add_inequality_constraint(nlopt_opt opt, nlopt_func fc, void* fc_data, double tol)
{
    return nlopt_add_precond_inequality_constraint(opt, fc, NULL, fc_data, tol);
}


nlopt_result LocalOptHelper::nlopt_add_inequality_mconstraint(nlopt_opt opt, unsigned m, nlopt_mfunc fc, void* fc_data, const double* tol)
{
    nlopt_result ret;
    //nlopt_unset_errmsg(opt);
    if (!m) {                   /* empty constraints are always ok */
        if (opt && opt->munge_on_destroy)
            opt->munge_on_destroy(fc_data);
        return NLOPT_SUCCESS;
    }
    if (!opt)
        ret = NLOPT_INVALID_ARGS;
    else if (!inequality_ok(opt->algorithm))
        ret = ERR(NLOPT_INVALID_ARGS, opt, "invalid algorithm for constraints");
    else
        ret = add_constraint(opt, &opt->m, &opt->m_alloc, &opt->fc, m, NULL, fc, NULL, fc_data, tol);
    if (ret < 0 && opt && opt->munge_on_destroy)
        opt->munge_on_destroy(fc_data);
    return ret;
}

int LocalOptHelper::equality_ok(nlopt_algorithm algorithm)
{
    /* equality constraints (h(x) = 0) only via some algorithms */
    return (algorithm == NLOPT_LD_SLSQP || algorithm == NLOPT_GN_ISRES || algorithm == NLOPT_LN_COBYLA);
}

nlopt_result LocalOptHelper::nlopt_add_precond_equality_constraint(nlopt_opt opt, nlopt_func fc, nlopt_precond pre, void* fc_data, double tol)
{
    nlopt_result ret;
    //nlopt_unset_errmsg(opt);
    if (!opt)
        ret = NLOPT_INVALID_ARGS;
    else if (!equality_ok(opt->algorithm))
        ret = ERR(NLOPT_INVALID_ARGS, opt, "invalid algorithm for constraints");
    else if (nlopt_count_constraints(opt->p, opt->h) + 1 > opt->n)
        ret = ERR(NLOPT_INVALID_ARGS, opt, "too many equality constraints");
    else
        ret = add_constraint(opt, &opt->p, &opt->p_alloc, &opt->h, 1, fc, NULL, pre, fc_data, &tol);
    if (ret < 0 && opt && opt->munge_on_destroy)
        opt->munge_on_destroy(fc_data);
    return ret;
}

nlopt_result LocalOptHelper::nlopt_add_equality_constraint(nlopt_opt opt, nlopt_func fc, void* fc_data, double tol)
{
    return nlopt_add_precond_equality_constraint(opt, fc, NULL, fc_data, tol);
}

nlopt_result LocalOptHelper::nlopt_remove_equality_constraints(nlopt_opt opt)
{
    unsigned i;
    //nlopt_unset_errmsg(opt);
    if (!opt)
        return NLOPT_INVALID_ARGS;
    if (opt->munge_on_destroy) {
        nlopt_munge munge = opt->munge_on_destroy;
        for (i = 0; i < opt->p; ++i)
            munge(opt->h[i].f_data);
    }
    for (i = 0; i < opt->p; ++i)
        free(opt->h[i].tol);
    free(opt->h);
    opt->h = NULL;
    opt->p = opt->p_alloc = 0;
    return NLOPT_SUCCESS;
}

nlopt_result LocalOptHelper::nlopt_add_equality_mconstraint(nlopt_opt opt, unsigned m, nlopt_mfunc fc, void* fc_data, const double* tol)
{
    nlopt_result ret;
    //nlopt_unset_errmsg(opt);
    if (!m) {                   /* empty constraints are always ok */
        if (opt && opt->munge_on_destroy)
            opt->munge_on_destroy(fc_data);
        return NLOPT_SUCCESS;
    }
    if (!opt)
        ret = NLOPT_INVALID_ARGS;
    else if (!equality_ok(opt->algorithm))
        ret = ERR(NLOPT_INVALID_ARGS, opt, "invalid algorithm for constraints");
    else if (nlopt_count_constraints(opt->p, opt->h) + m > opt->n)
        ret = ERR(NLOPT_INVALID_ARGS, opt, "too many equality constraints");
    else
        ret = add_constraint(opt, &opt->p, &opt->p_alloc, &opt->h, m, NULL, fc, NULL, fc_data, tol);
    if (ret < 0 && opt && opt->munge_on_destroy)
        opt->munge_on_destroy(fc_data);
    return ret;
}
