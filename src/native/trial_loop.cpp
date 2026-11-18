// The training trial loop of src/fast_trial.py (_run, _Trial.control,
// _Trial.should_stop), in C++. Results are bit-identical to the Python loop.
//
// "Bit-identical" is kept the same way fast_trial.py keeps it: every
// floating-point operation is done in the same order on the same values, and
// every operation whose rounding is not fixed by IEEE 754 goes through the
// very routine Python uses, handed over as a function pointer by
// src/native_trial.py:
//
//   np.matmul (CPG step)   -> NumPy's OpenBLAS cblas_dgemv
//   np.dot (steering)      -> NumPy's OpenBLAS cblas_ddot
//   np.tanh                -> NumPy's own float64 tanh ufunc loop
//   math.sin/cos/atan2, ** -> the C library Python's math module uses
//
// Plain + - * / and comparisons are exactly rounded, so they need no special
// care, as long as the compiler does not fuse them: build with
// -ffp-contract=off and without -ffast-math (see build_native.py).
//
// MuJoCo is called through function pointers too, so this file needs only
// MuJoCo's headers (for the mjModel / mjData layout), not its import library.

#include <cmath>
#include <cstdint>
#include <cstring>

#include <mujoco/mujoco.h>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

typedef void (*mj_fn)(const mjModel*, mjData*);
typedef double (*fn1)(double);
typedef double (*fn2)(double, double);
typedef double (*ddot_fn)(int64_t, const double*, int64_t, const double*, int64_t);
typedef void (*dgemv_fn)(int, int, int64_t, int64_t, double, const double*, int64_t,
                         const double*, int64_t, double, double*, int64_t);
typedef void (*ufunc_loop_fn)(char**, const intptr_t*, const intptr_t*, void*);

// Everything the loop needs, filled in by src/native_trial.py.
struct Funcs {
    mj_fn mj_step;
    mj_fn mj_forward;
    fn1 sin_;
    fn1 cos_;
    fn2 atan2_;
    fn2 pow_;
    ddot_fn ddot;
    dgemv_fn dgemv;
    ufunc_loop_fn tanh_loop;
    void* tanh_data;
};

struct Spec {
    double control_step;
    double sample_step;
    double end;  // simulation_time, or +inf
    int32_t robot_id;
    int32_t ball_id;
    // CPG
    int32_t n_state;
    double* state;          // n_state, updated in place
    const double* weights;  // n_state x n_state, C order
    double* work;           // 5 * n_state scratch
    // Outputs (one per hinge)
    int32_t n_out;
    const int32_t* state_index;  // n_out
    const double* hinge_range;   // n_out
    const double* steering;      // n_out x n_in, C order
    int32_t n_in;                // steering inputs (4)
    const int32_t* pos_index;    // n_out
    const int32_t* vel_index;    // n_out
    double steering_scale;
    double distance_scale;
    int32_t brain_has_offset;
    double brain_offset[3];
    // Stop conditions
    int32_t n_stops;
    const int32_t* stop_robot;
    const int32_t* stop_object;
    const double* stop_distance;
    const int32_t* stop_has_offset;
    const double* stop_offset;  // n_stops x 3
    // Samples (output)
    int32_t max_samples;
    double* robot_pos;   // max_samples x 3
    double* robot_quat;  // max_samples x 4
    double* ball_pos;    // max_samples x 3
};

namespace {

const double PI = 3.141592653589793;  // math.pi

// Python's max(a, b) / min(a, b) for two floats: the first argument unless the
// second compares greater / smaller (so a NaN first argument is returned).
inline double py_max(double a, double b) { return b > a ? b : a; }
inline double py_min(double a, double b) { return b < a ? b : a; }

// np.maximum / np.minimum (NaN propagating).
inline double np_maximum(double a, double b) { return (a >= b || std::isnan(a)) ? a : b; }
inline double np_minimum(double a, double b) { return (a <= b || std::isnan(a)) ? a : b; }

// Python's float % float.
inline double py_mod(double x, double y) {
    double mod = std::fmod(x, y);
    if (mod != 0.0) {
        if ((y < 0.0) != (mod < 0.0)) mod += y;
    } else {
        mod = std::copysign(0.0, y);
    }
    return mod;
}

// ball_aware_brain._wrap_angle
inline double wrap_angle(double angle) { return py_mod(angle + PI, 2.0 * PI) - PI; }

// fast_trial._rotate: pyrr's cross(q, cross(v4, conjugate(q))), term for term.
inline void rotate(double qx, double qy, double qz, double qw, double vx, double vy, double vz,
                   double* ox, double* oy, double* oz) {
    const double cx = -qx, cy = -qy, cz = -qz, cw = qw;
    const double vw = 0.0;
    const double ix = vx * cw + vy * cz - vz * cy + vw * cx;
    const double iy = -vx * cz + vy * cw + vz * cx + vw * cy;
    const double iz = vx * cy - vy * cx + vz * cw + vw * cz;
    const double iw = -vx * cx - vy * cy - vz * cz + vw * cw;
    *ox = qx * iw + qy * iz - qz * iy + qw * ix;
    *oy = -qx * iz + qy * iw + qz * ix + qw * iy;
    *oz = qx * iy - qy * ix + qz * iw + qw * iz;
}

// fast_trial._reference_point: the core, or the core plus the offset rotated by
// the physical orientation (MuJoCo's xquat is w, x, y, z).
inline void reference_point(const double* pos, const double* quat, int has_offset,
                            const double* offset, double* rx, double* ry, double* rz) {
    if (!has_offset) {
        *rx = pos[0];
        *ry = pos[1];
        *rz = pos[2];
        return;
    }
    double ox, oy, oz;
    rotate(quat[1], quat[2], quat[3], quat[0], offset[0], offset[1], offset[2], &ox, &oy, &oz);
    *rx = pos[0] + ox;
    *ry = pos[1] + oy;
    *rz = pos[2] + oz;
}

struct Loop {
    const Funcs& f;
    const Spec& s;
    const mjModel* m;
    mjData* d;
    int32_t n_samples = 0;

    double tanh(double x) const {
        double in = x, out = 0.0;
        char* args[2] = {reinterpret_cast<char*>(&in), reinterpret_cast<char*>(&out)};
        const intptr_t dims[1] = {1};
        const intptr_t steps[2] = {sizeof(double), sizeof(double)};
        f.tanh_loop(args, dims, steps, f.tanh_data);
        return out;
    }

    // np.matmul(weights, v) for a C-ordered square matrix and a vector.
    void matmul(const double* v, double* out) const {
        const int64_t n = s.n_state;
        f.dgemv(101 /*CblasRowMajor*/, 111 /*CblasNoTrans*/, n, n, 1.0, s.weights, n, v, 1, 0.0,
                out, 1);
    }

    // fast_trial._rk45_exact
    void rk45(double dt) {
        const int n = s.n_state;
        double* state = s.state;
        double* a1 = s.work;
        double* a2 = a1 + n;
        double* a3 = a2 + n;
        double* a4 = a3 + n;
        double* tmp = a4 + n;
        const double half = dt / 2.0;
        matmul(state, a1);
        for (int i = 0; i < n; ++i) tmp[i] = state[i] + half * a1[i];
        matmul(tmp, a2);
        for (int i = 0; i < n; ++i) tmp[i] = state[i] + half * a2[i];
        matmul(tmp, a3);
        for (int i = 0; i < n; ++i) tmp[i] = state[i] + dt * a3[i];
        matmul(tmp, a4);
        const double sixth = dt / 6.0;
        for (int i = 0; i < n; ++i) {
            const double sum = a1[i] + 2.0 * (a2[i] + a3[i]) + a4[i];
            const double next = state[i] + sixth * sum;
            state[i] = np_minimum(np_maximum(next, -1.0), 1.0);
        }
    }

    // fast_trial._Trial.control
    void control(double dt) {
        const double* robot = d->xpos + 3 * s.robot_id;
        const double* quat = d->xquat + 4 * s.robot_id;
        const double* ball = d->xpos + 3 * s.ball_id;

        rk45(dt);

        double rx = robot[0], ry = robot[1], rz = robot[2];
        double q[4];
        if (s.brain_has_offset) {
            reference_point(robot, quat, 1, s.brain_offset, &rx, &ry, &rz);
            q[0] = quat[1]; q[1] = quat[2]; q[2] = quat[3]; q[3] = quat[0];
        } else {
            q[0] = quat[0]; q[1] = quat[1]; q[2] = quat[2]; q[3] = quat[3];
        }
        const double to_x = ball[0] - rx;
        const double to_y = ball[1] - ry;
        const double distance = std::sqrt(f.pow_(to_x, 2.0) + f.pow_(to_y, 2.0));
        const double target_angle = f.atan2_(to_y, to_x);
        double hx, hy, hz;
        rotate(q[0], q[1], q[2], q[3], 1.0, 0.0, 0.0, &hx, &hy, &hz);
        const double heading_angle = f.atan2_(hy, hx);
        const double angle_error = wrap_angle(target_angle - heading_angle);

        double inputs[8];
        inputs[0] = f.sin_(angle_error);
        inputs[1] = f.cos_(angle_error);
        inputs[2] = py_min(py_max(distance / s.distance_scale, 0.0), 1.0);
        inputs[3] = 1.0;

        for (int k = 0; k < s.n_out; ++k) {
            const double r = s.hinge_range[k];
            const double cpg_target = s.state[s.state_index[k]] * r;
            // np.dot: DOUBLE_dot sums the BLAS result into 0.0.
            double dot = 0.0;
            dot += f.ddot(s.n_in, s.steering + static_cast<int64_t>(k) * s.n_in, 1, inputs, 1);
            const double steering_fraction = tanh(dot) * s.steering_scale;
            const double steering_offset = steering_fraction * r;
            d->ctrl[s.pos_index[k]] = py_min(py_max(cpg_target + steering_offset, -r), r);
            d->ctrl[s.vel_index[k]] = 0.0;
        }
    }

    // fast_trial._Trial.should_stop
    bool should_stop() const {
        for (int i = 0; i < s.n_stops; ++i) {
            const double* robot = d->xpos + 3 * s.stop_robot[i];
            const double* quat = d->xquat + 4 * s.stop_robot[i];
            const double* obj = d->xpos + 3 * s.stop_object[i];
            double rx, ry, rz;
            reference_point(robot, quat, s.stop_has_offset[i], s.stop_offset + 3 * i, &rx, &ry, &rz);
            const double distance = std::sqrt(f.pow_(rx - obj[0], 2.0) + f.pow_(ry - obj[1], 2.0));
            if (distance <= s.stop_distance[i]) return true;
        }
        return false;
    }

    bool sample() {
        if (n_samples >= s.max_samples) return false;
        std::memcpy(s.robot_pos + 3 * n_samples, d->xpos + 3 * s.robot_id, 3 * sizeof(double));
        std::memcpy(s.robot_quat + 4 * n_samples, d->xquat + 4 * s.robot_id, 4 * sizeof(double));
        std::memcpy(s.ball_pos + 3 * n_samples, d->xpos + 3 * s.ball_id, 3 * sizeof(double));
        ++n_samples;
        return true;
    }

    // fast_trial._run. Returns the sample count, or a negative error code.
    int32_t run() {
        const double control_step = s.control_step;
        const double sample_step = s.sample_step;
        const double end = s.end;
        const double timestep = m->opt.timestep;

        f.mj_forward(m, d);
        if (!sample()) return -2;

        double last_control_time = 0.0;
        double last_sample_time = 0.0;
        double time = d->time;
        while (time < end) {
            if (time >= last_control_time + control_step) {
                last_control_time = std::floor(time / control_step) * control_step;
                control(control_step);
                if (should_stop()) {
                    if (!sample()) return -2;
                    break;
                }
            }
            if (time >= last_sample_time + sample_step) {
                last_sample_time = std::trunc(time / sample_step) * sample_step;
                if (!sample()) return -2;
            }

            const double next_control = last_control_time + control_step;
            const double next_sample = last_sample_time + sample_step;
            double predicted = time;
            int steps = 0;
            while (true) {
                predicted = predicted + timestep;
                ++steps;
                if (!(predicted < end) || predicted >= next_control || predicted >= next_sample) break;
            }
            for (int i = 0; i < steps; ++i) f.mj_step(m, d);
            time = d->time;
            if (time != predicted) return -1;
        }
        if (!sample()) return -2;
        return n_samples;
    }
};

}  // namespace

EXPORT int32_t trial_loop_abi_version() { return 1; }

// The MuJoCo version whose headers (mjModel / mjData layout) this was built
// with. src/native_trial.py refuses to run against any other version.
EXPORT int32_t trial_loop_mujoco_version() { return mjVERSION_HEADER; }

EXPORT int32_t run_trial(const Funcs* funcs, const Spec* spec, const mjModel* m, mjData* d) {
    Loop loop{*funcs, *spec, m, d};
    return loop.run();
}
