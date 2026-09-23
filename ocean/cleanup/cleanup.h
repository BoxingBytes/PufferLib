// Clean Up -- C port of SocialJax's clean_up.py
//
// Deliberate divergences from the SocialJax reference (Most are bugs there):
//   1. Actions are EGOCENTRIC (w.r.t agent_heading). SocialJax's STEP_MOVE ignores agent heading
//   2. No inventory. SocialJax's inv system is never used, same for freeze
//   3. Beam targets are bounds-safe. In socialJax, an agent facing the
//      grid edge & fires kills itself.
//   4. Observations are not one-hot. One-hot / embedding is the
//      policy's job.
//   5. Individual reward, 1.0 per apple, no scaling by num_agents, and no
//      zeroing of the reward on the episode's final step.
//   6. Entities and water live in two disjoint layers. SocialJax stamps both
//      into one grid, so an agent hides the dirt it stands on.
//   7. Cleaning ignores occupancy. SocialJax tests the composite grid, so an
//      agent body-blocks the dirt under it.
//   8. Beam markers draw over water. SocialJax's clean beam is invisible over
//      the river, the one place it does anything.

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"

#define MAX_AGENTS 16
#define HORIZON 1000 // SocialJax's num_inner_steps

// ---------------------------------------------------------------- grid codes
#define EMPTY       0
#define WALL        1   // padding border only; the map has no walls
#define BEAM        2
#define CLEAN_BEAM  3
#define APPLE       4
#define AGENT_BASE  5   // cell holds AGENT_BASE + agent_id

// ------------------------------------------------------------- terrain codes
#define T_EMPTY     0
#define T_RIVER     1   // inert: never dirty, never cleanable
#define T_POT_DIRT  2   // clean water, can become dirty
#define T_DIRT      3

// ------------------------------------------------------------------- actions
#define ACTION_TURN_LEFT     0
#define ACTION_TURN_RIGHT    1
#define ACTION_FORWARD       2
#define ACTION_BACKWARD      3
#define ACTION_STRAFE_LEFT   4
#define ACTION_STRAFE_RIGHT  5
#define ACTION_NOOP          6
#define ACTION_ZAP           7
#define ACTION_CLEAN         8
#define NUM_ACTIONS          9

// ---------------------------------------------------------------- directions
#define NUM_DIRS 4
static const int8_t DIR_DROW[NUM_DIRS] = {1, 0, -1, 0};
static const int8_t DIR_DCOL[NUM_DIRS] = {0, 1, 0, -1};

// ------------------------------------------------------------------ regrowth
#define MAX_APPLE_GROWTH_RATE 0.05f
#define THRESHOLD_DEPLETION   0.4f

// ------------------------------------------------------------- dirt dynamics
#define DIRT_SPAWN_PROB  0.5f
#define DIRT_SPAWN_DELAY 50 // Timesteps before dirt can spawn. 

// --------------------------------------------------------------------- beams
#define BEAM_MASK_SIZE 4
// The "T" shape in agent heading coordinates, shared by both beams.
static const int8_t BEAM_DR[BEAM_MASK_SIZE] = {+1, +1, +1, +2};
static const int8_t BEAM_DC[BEAM_MASK_SIZE] = { 0, -1, +1,  0};

// ----------------------------------------------------------------- obs shape
#define NUM_OBS_CHANNELS 3
// SocialJax originally one-hot encode into 15 channels.
// Most of which were useless (inventory/frozen..). For the rest we let the policy one-hot.
// plane 0 -- entity: EMPTY / WALL / BEAM / CLEAN_BEAM / APPLE / 5 = an agent
// plane 1 -- terrain: T_EMPTY / T_RIVER / T_POT_DIRT / T_DIRT
// plane 2 -- agent detail: 0 where no agent, else 1 + 4*id + rel_dir if
//            differentiate_other_agents_in_obs, else 1 + rel_dir.
//            rel_dir = (their_dir - my_dir) & 3.
#define OBS_WINDOW 11

// ------------------------------------------------------------- map constants
#define MAP_HEIGHT 19
#define MAP_WIDTH  28
#define N_APPLE       122  // 'A'
#define N_DIRT_CELLS  147  // 'C' + 'D'
#define N_INIT_DIRT    79  // 'D'
#define N_INIT_CLEAN   68  // 'C'
#define N_RIVER        20  // 'R'
#define N_SPAWN        19  // 'P'

// 20 River cells that never gets dirty
#define DIRT_DENOM (N_DIRT_CELLS + N_RIVER)

typedef struct Log {
    float time_to_growth;  // tick dirt_fraction first drops below the threshold
    float mean_dirt_fraction;
    float sustainability;
    float efficiency;
    float clean_rate;      // agent-steps choosing ACTION_CLEAN
    float clean_hit_rate;  // cells converted per agent-step
    float zap_rate;
    float hit_rate;
    float zap_timing;
    float hit_timing;
    float equality;
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Client Client;

typedef struct Env Env;
struct Env {
    // ------------------------------------------------ PufferLib interface
    uint8_t* observations;  // (num_agents, NUM_OBS_CHANNELS, OBS_WINDOW, OBS_WINDOW)
    float* actions;         // (num_agents,)
    float* rewards;         // (num_agents,)
    float* terminals;       // (num_agents,)
    int num_agents;
    Log log;
    Client* client;

    // ---------------------------------------------------------- geometry
    // All stored indices are PADDED flat indices; only the renderer converts back.
    int ascii_height;    // map_height, UNPADDED, from ASCII map
    int ascii_width;     // map_width, UNPADDED, from ASCII map
    int pad;       // OBS_WINDOW - 1
    int stride;    // map_width + 2*pad

    uint8_t* grid;     // entities; border filled with WALL
    uint8_t* terrain;  // water; border T_EMPTY

    // ---------------------------------------------------------- observations
    int32_t start[NUM_DIRS];       // flat relative index of the "behind to the left" corner
    int32_t row_stride[NUM_DIRS];  // flat relative index of the next row in the obs window
    int32_t col_stride[NUM_DIRS];  // flat relative index of the next col in the obs window

    // ------------------------------------------------------------ config
    bool differentiate_other_agents_in_obs;  // default true
    bool shared_rewards;          // default true (SocialJax's default here)

    // ------------------------------------------------------------ agents
    int32_t *agents_idx;  // padded flat cell index
    uint8_t *agents_dir;  // 0..3

    // ---------------------------------------------------- step scratch
    int32_t *target;  // proposed then resolved destination cell
    uint8_t *hit;     // (num_agents,) FLAGS, consumed by apply_respawns next step

    // ------------------------------------------------- static cell sets
    // Parsed once from the ASCII map (see init). Padded flat indices, kept so
    // both bag refills at reset are a memcpy.
    int32_t *apple_idx;       // 'A'
    int32_t *init_dirt_idx;   // 'D'
    int32_t *init_clean_idx;  // 'C'
    int32_t *river_idx;       // 'R', stamped at reset only
    int32_t *respawn_idx;     // 'P'

    // ----------------------------------------------------------- bags
    // Unordered. For iteration & sampling speed
    int32_t *clean_idx;       // T_POT_DIRT cells, N_DIRT_CELLS capacity
    int n_clean;
    int32_t *dead_apple_idx;  // empty apple cells, N_APPLE capacity
    int n_dead;

    float dirt_fraction;   // (N_DIRT_CELLS - n_clean) / DIRT_DENOM, per step
    float apple_growth_p;  // derived from dirt_fraction, per step

    // -------------------------------------------------------------- beams
    int32_t *beam_idx;   // Write beamed cells. For faster clearing lookup only
    int n_beam;          // 4*num_agents: an agent fires at most one beam per step

    // ---------------------------------------------------------------- rng
    uint32_t rng;  // Seed in, state out: set before init(), which sanitizes it. rand() gave issues with low probs

    int tick;

    // --------------------------------------------------------------- map
    const char *map_ASCII[MAP_HEIGHT];

    // ----------------------------------------------------------- logging
    float sum_ticks_apple_collected; // For computing sustainability
    float tot_apple_collected;
    float sum_dirt_fraction;         // For computing mean_dirt_fraction
    float tot_cleans;                // ACTION_CLEAN chosen
    float tot_cells_cleaned;         // cells actually converted
    float tot_zaps;
    float tot_hits;
    float sum_ticks_zap;             // For computing zap_timing
    float sum_ticks_hit;             // For computing hit_timing
    float tick_growth;               // Tick dirt_fraction first dropped below the threshold, 0 if never
    float *agent_returns;            // For computing equality
};

// ================================================================== rng
// x ^= x<<13; x ^= x>>17; x ^= x<<5.
static inline uint32_t rnd(Env* env){
    uint32_t x = env->rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    env->rng = x;
    return x;
}

// Uniform [0, 1): (rnd() >> 8) * (1.0f / 16777216.0f).
static inline float rndf(Env* env){
    return (rnd(env) >> 8) * (1.0f / 16777216.0f);
}

// Reseeds. Substitutes 1 for a zero seed.
void c_seed(Env* env, uint32_t seed){
    env->rng = (seed == 0) ? 1 : seed;
}

void init(Env* env){
    c_seed(env, env->rng);
    const char *map[MAP_HEIGHT] = {
        "CDDDCDDCDCDCDCDCDCDCCDCDDDCD",
        "CDCDCDDCDCDCDCDCDCDCCDCDDDCD",
        "CDDCDDCCDCDCDCDCDCDCCDCDDDCD",
        "CDCDCDDCDCDCDCDCDCDCCDCDDDCD",
        "CDDDDDDCDCDCDCDCDCDCCDCDDDCD",
        "               RDCCCCCC     ",
        "   P    P          RRR      ",
        "     P     P   P   RR   P   ",
        "             P   P RR       ",
        "   P    P          RR    P  ",
        "               P   RR P     ",
        "     P           P RR       ",
        "           P       RR  P    ",
        "  P             P  RR       ",
        " A A A A A A A A A RR  A A A",
        "AAAAAAAAAAAAAAAAAAA  AAAAAAA",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    };
    memcpy(env->map_ASCII, map, sizeof(map));

    env->log = (Log){0};

    env->ascii_width = strlen(map[0]);
    env->ascii_height = sizeof(map) / sizeof(map[0]);
    env->pad = OBS_WINDOW - 1; //Because obs start one cell behind the agent
    env->stride = env->ascii_width + 2*env->pad;
    assert(env->pad >= 2);
    assert(env->num_agents > 0 && env->num_agents <= MAX_AGENTS);

    // Observation windows sweep from "left-to-right" relative to agent dir, from behind to ahead.
    for (int d = 0; d < NUM_DIRS; d++){
        env->row_stride[d] = DIR_DROW[d]*env->stride + DIR_DCOL[d];
        env->col_stride[d] = DIR_DCOL[d]*env->stride - DIR_DROW[d];
        env->start[d] = -env->row_stride[d] - (OBS_WINDOW/2)*env->col_stride[d];
    }

    const int tot_flattened = (env->ascii_height + 2*env->pad)*(env->stride);
    env->grid = (uint8_t*)calloc(tot_flattened, sizeof(uint8_t));
    memset(env->grid, WALL, tot_flattened*sizeof(uint8_t));
    env->terrain = (uint8_t*)calloc(tot_flattened, sizeof(uint8_t));

    env->apple_idx = (int32_t*)calloc(N_APPLE, sizeof(int32_t));
    env->init_dirt_idx = (int32_t*)calloc(N_INIT_DIRT, sizeof(int32_t));
    env->init_clean_idx = (int32_t*)calloc(N_INIT_CLEAN, sizeof(int32_t));
    env->river_idx = (int32_t*)calloc(N_RIVER, sizeof(int32_t));
    env->respawn_idx = (int32_t*)calloc(N_SPAWN, sizeof(int32_t));

    env->clean_idx = (int32_t*)calloc(N_DIRT_CELLS, sizeof(int32_t));
    env->n_clean = 0;
    env->dead_apple_idx = (int32_t*)calloc(N_APPLE, sizeof(int32_t));
    env->n_dead = 0;

    env->agents_idx = (int32_t*)calloc(env->num_agents, sizeof(int32_t));
    env->agents_dir = (uint8_t*)calloc(env->num_agents, sizeof(uint8_t));
    env->target = (int32_t*)calloc(env->num_agents, sizeof(int32_t));
    env->hit = (uint8_t*)calloc(env->num_agents, sizeof(uint8_t));
    env->beam_idx = (int32_t*)calloc(BEAM_MASK_SIZE*env->num_agents, sizeof(int32_t));
    env->n_beam = 0;

    const int offset = env->stride*env->pad + env->pad;
    int n_apple = 0, n_dirt = 0, n_clean = 0, n_river = 0, n_spawn = 0;

    for (int r = 0; r < env->ascii_height; r++){
        for (int c = 0; c < env->ascii_width; c++){
            const char cell = map[r][c];
            const int idx = offset + r*env->stride + c;
            env->grid[idx] = EMPTY;
            switch (cell){
                case 'A':
                    env->apple_idx[n_apple++] = idx;
                    break;
                case 'D':
                    env->init_dirt_idx[n_dirt++] = idx;
                    break;
                case 'C':
                    env->init_clean_idx[n_clean++] = idx;
                    break;
                case 'R':
                    env->river_idx[n_river++] = idx;
                    break;
                case 'P':
                    env->respawn_idx[n_spawn++] = idx;
                    break;
                default:
                    break;
            }
        }
    };

    // Just making sure cuz those were eyeballed
    assert(n_apple == N_APPLE);
    assert(n_dirt == N_INIT_DIRT);
    assert(n_clean == N_INIT_CLEAN);
    assert(n_dirt + n_clean == N_DIRT_CELLS);
    assert(n_river == N_RIVER);
    assert(n_spawn == N_SPAWN);

    // Logging
    env->agent_returns = (float*)calloc(env->num_agents, sizeof(float));
};

void compute_observations(Env* env){
    const int32_t plane_size = OBS_WINDOW*OBS_WINDOW;

    for (int a = 0; a < env->num_agents; a++){
        const int32_t channel0_base = a*NUM_OBS_CHANNELS*plane_size;
        const int32_t channel1_base = channel0_base + plane_size;
        const int32_t channel2_base = channel1_base + plane_size;

        const int32_t agent_idx = env->agents_idx[a];
        const uint8_t agent_dir = env->agents_dir[a];
        // Locate "behind to the left" corner.
        const int32_t start_idx = agent_idx + env->start[agent_dir];

        for (int r = 0; r < OBS_WINDOW; r++){
            for (int c = 0; c < OBS_WINDOW; c++){
                // Then use precomputed row/col strides to walk the window, filling in the values.
                const int32_t cell_idx = (start_idx + r*env->row_stride[agent_dir]) + c*env->col_stride[agent_dir];
                uint8_t cell_content = env->grid[cell_idx];
                const int32_t k = r*OBS_WINDOW + c;

                if (cell_content >= AGENT_BASE){
                    const int32_t other_agent_id = cell_content - AGENT_BASE;
                    const uint8_t other_agent_dir = env->agents_dir[other_agent_id];
                    const int32_t rel_dir = (other_agent_dir - agent_dir) & 3; // works because modulo is 2^n and rel_dir can be < 0

                    if (env->differentiate_other_agents_in_obs){
                        // Encoding both agent_id & direction in one go, bad idea?
                        env->observations[channel2_base + k] = 1 + 4*other_agent_id + rel_dir;
                    } else {
                        env->observations[channel2_base + k] = 1 + rel_dir;
                    }

                    cell_content = AGENT_BASE; // For channel 0
                }
                env->observations[channel0_base + k] = cell_content;
                env->observations[channel1_base + k] = env->terrain[cell_idx];
            }
        }
    };
};

void update_dirt_fraction(Env* env){
    const int n_dirt = N_DIRT_CELLS - env->n_clean;
    env->dirt_fraction = n_dirt / (float)DIRT_DENOM;
    // Goes negative past the threshold, which is fine: rndf() >= 0
    env->apple_growth_p = MAX_APPLE_GROWTH_RATE*(1.0f - env->dirt_fraction/THRESHOLD_DEPLETION);
};

void c_reset(Env* env){
    env->tick = 0;
    memset(env->hit, 0, env->num_agents*sizeof(uint8_t));
    memset(env->target, 0, env->num_agents*sizeof(int32_t));
    memset(env->beam_idx, 0, BEAM_MASK_SIZE*env->num_agents*sizeof(int32_t));
    env->n_beam = 0;

    // Puffer
    memset(env->observations, 0, env->num_agents*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW*sizeof(uint8_t));

    // Rebuild both layers
    const int offset = env->stride*env->pad + env->pad;
    for (int r = 0; r < env->ascii_height; r++){
        memset(env->grid + offset + r*env->stride, EMPTY, env->ascii_width*sizeof(uint8_t));
        memset(env->terrain + offset + r*env->stride, T_EMPTY, env->ascii_width*sizeof(uint8_t));
    }
    for (int i = 0; i < N_RIVER; i++){
        env->terrain[env->river_idx[i]] = T_RIVER;
    }
    for (int i = 0; i < N_INIT_DIRT; i++){
        env->terrain[env->init_dirt_idx[i]] = T_DIRT;
    }
    for (int i = 0; i < N_INIT_CLEAN; i++){
        env->terrain[env->init_clean_idx[i]] = T_POT_DIRT;
    }

    memcpy(env->clean_idx, env->init_clean_idx, N_INIT_CLEAN*sizeof(int32_t));
    env->n_clean = N_INIT_CLEAN;
    memcpy(env->dead_apple_idx, env->apple_idx, N_APPLE*sizeof(int32_t));
    env->n_dead = N_APPLE;
    update_dirt_fraction(env);

    for (int i = 0; i < env->num_agents; i++){
        int32_t idx;
        do {
            idx = env->respawn_idx[rnd(env) % N_SPAWN];
        } while (env->grid[idx] != EMPTY);
        env->grid[idx] = AGENT_BASE + i;
        env->agents_idx[i] = idx;
        env->agents_dir[i] = rnd(env) % NUM_DIRS;
    }

    // Logging
    env->sum_ticks_apple_collected = 0.0f;
    env->tot_apple_collected = 0.0f;
    env->sum_dirt_fraction = 0.0f;
    env->tot_cleans = 0.0f;
    env->tot_cells_cleaned = 0.0f;
    env->tot_zaps = 0.0f;
    env->tot_hits = 0.0f;
    env->sum_ticks_zap = 0.0f;
    env->sum_ticks_hit = 0.0f;
    env->tick_growth = 0.0f;
    memset(env->agent_returns, 0, env->num_agents*sizeof(float));

    compute_observations(env);
};

// Uses the Gini coefficient as in inequity aversion (Hughes et. al 2018)
float compute_equality(Env* env){
    float double_sum = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < env->num_agents; i++){
        sum += env->agent_returns[i];
        for (int j = 0; j < env->num_agents; j++){
            double_sum += fabsf(env->agent_returns[i] - env->agent_returns[j]);
        }
    }
    if (sum == 0.0f) return 1.0f; // All agents have zero return, perfect equality
    return 1.0f - double_sum / (2.0f * env->num_agents * sum);
};

void add_log(Env* env){
    const float agent_steps = env->num_agents*(float)HORIZON;

    env->log.time_to_growth += (env->tick_growth == 0.0f) ? (float)HORIZON : env->tick_growth;
    env->log.mean_dirt_fraction += env->sum_dirt_fraction / (float)HORIZON;

    if (env->tot_apple_collected == 0.0f){
        env->log.sustainability += (float)HORIZON;
    } else {
        env->log.sustainability += env->sum_ticks_apple_collected / env->tot_apple_collected;
    }

    if (env->tot_zaps == 0.0f){
        env->log.zap_timing += (float)HORIZON;
    } else {
        env->log.zap_timing += env->sum_ticks_zap / env->tot_zaps;
    }

    if (env->tot_hits == 0.0f){
        env->log.hit_timing += (float)HORIZON;
    } else {
        env->log.hit_timing += env->sum_ticks_hit / env->tot_hits;
    }

    env->log.equality += compute_equality(env);
    env->log.efficiency += env->tot_apple_collected / (float)HORIZON;
    env->log.clean_rate += env->tot_cleans / agent_steps;
    env->log.clean_hit_rate += env->tot_cells_cleaned / agent_steps;
    env->log.zap_rate += env->tot_zaps / agent_steps;
    env->log.hit_rate += env->tot_hits / agent_steps;

    env->log.perf += env->tot_apple_collected / agent_steps; // Loose
    env->log.score += env->tot_apple_collected;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->tot_apple_collected;

    env->log.n++;
};

void apply_respawns(Env* env){
    for (int a = 0; a < env->num_agents; a++){
        if (!env->hit[a]) continue;
        env->hit[a] = 0;
        int32_t respawn_cell;
        do {
            respawn_cell = env->respawn_idx[rnd(env) % N_SPAWN];
        } while (env->grid[respawn_cell] != EMPTY);
        env->grid[env->agents_idx[a]] = EMPTY;
        env->agents_idx[a] = respawn_cell;
        env->agents_dir[a] = (rnd(env) % (NUM_DIRS - 1)) + 1; // 1..3, never 0 = original SocialJax
        env->grid[respawn_cell] = AGENT_BASE + a;
    }
};

// Every apple has apple_regrow_p chance to regrow. 
void apple_regrow(Env* env){
    if (env->apple_growth_p <= 0.0f) return;

    for (int i = env->n_dead - 1; i >= 0; i--){
        const int32_t cell = env->dead_apple_idx[i];
        if (env->grid[cell] != EMPTY) continue;
        if (rndf(env) >= env->apple_growth_p) continue;
        env->grid[cell] = APPLE;
        env->dead_apple_idx[i] = env->dead_apple_idx[--env->n_dead];
    }
};

// At most one T_POT_DIRT cell becomes T_DIRT per step, chosen uniformly,
// accepted with DIRT_SPAWN_PROB, and only once tick > DIRT_SPAWN_DELAY.
void spawn_dirt(Env* env){
    if (env->n_clean == 0) return;
    if (env->tick <= DIRT_SPAWN_DELAY) return;
    if (rndf(env) >= DIRT_SPAWN_PROB) return;

    const int i = rnd(env) % env->n_clean;
    const int32_t cell = env->clean_idx[i];
    env->clean_idx[i] = env->clean_idx[--env->n_clean];
    env->terrain[cell] = T_DIRT;
};

void beam_clear(Env* env){
    for (int i = 0; i < env->n_beam; i++){
        const int32_t cell = env->beam_idx[i];
        if (env->grid[cell] == BEAM || env->grid[cell] == CLEAN_BEAM) env->grid[cell] = EMPTY;
    }
    env->n_beam = 0;
};

void compute_targets(Env* env){
    memset(env->target, 0, env->num_agents*sizeof(int32_t));
    for (int a = 0; a < env->num_agents; a++){
        const int32_t agent_idx = env->agents_idx[a];
        const uint8_t agent_dir = env->agents_dir[a];
        const int32_t action = (int32_t)env->actions[a];

        int32_t target_idx = agent_idx;
        uint8_t new_dir = agent_dir;

        switch (action){
            case ACTION_TURN_LEFT:
                new_dir = (agent_dir + 1) & 3;
                break;
            case ACTION_TURN_RIGHT:
                new_dir = (agent_dir - 1) & 3;
                break;
            case ACTION_FORWARD:
                target_idx += env->row_stride[agent_dir];
                break;
            case ACTION_BACKWARD:
                target_idx -= env->row_stride[agent_dir];
                break;
            case ACTION_STRAFE_LEFT:
                target_idx -= env->col_stride[agent_dir];
                break;
            case ACTION_STRAFE_RIGHT:
                target_idx += env->col_stride[agent_dir];
                break;
            case ACTION_NOOP:
            case ACTION_ZAP:
            case ACTION_CLEAN:
                break;
            default:
                printf("compute_targets: invalid action %d\n", action);
                break;
        }

        if (env->grid[target_idx] == WALL) target_idx = agent_idx;

        env->target[a] = target_idx;
        env->agents_dir[a] = new_dir;
    }
};

// Agent-agent conflict resolution, Melting Pot semantics (see
// socialjax/environments/movement.py): a non-mover always keeps its cell;
// among movers contesting one cell a uniformly random winner is chosen and the
// rest revert; 2-cycles (swaps) are blocked.
void resolve_conflicts(Env* env){
    bool conflict = false;
    uint8_t n_iter = 0;
    do {
        for (int i = 0; i < env->num_agents; i++){
            for (int j = i + 1; j < env->num_agents; j++){

                // Swaps btw 2 agents disabled
                if (env->target[i] == env->agents_idx[j] && env->target[j] == env->agents_idx[i]){
                    env->target[i] = env->agents_idx[i];
                    env->target[j] = env->agents_idx[j];
                    conflict = true;
                    continue;
                }

                if (env->target[i] == env->target[j]){
                    conflict = true;

                    // Non mover always keeps its cell
                    if (env->target[i] == env->agents_idx[i]){
                        env->target[j] = env->agents_idx[j];
                        continue;
                    } else if (env->target[j] == env->agents_idx[j]){
                        env->target[i] = env->agents_idx[i];
                        continue;
                    }

                    // Randomly pick a winner, revert the loser to its original cell.
                    if (rnd(env) & 1){
                        env->target[j] = env->agents_idx[j];
                    } else {
                        env->target[i] = env->agents_idx[i];
                    }
                }
            }
        }
        n_iter++;
    } while (conflict && n_iter < env->num_agents - 1);
};

void collect_apples(Env* env){
    int n_collected = 0;
    for (int a = 0; a < env->num_agents; a++){
        const int32_t target_idx = env->target[a];
        if (env->grid[target_idx] != APPLE) continue;
        if (!env->shared_rewards) env->rewards[a] += 1.0f;
        n_collected++;
        env->grid[target_idx] = EMPTY;
        env->dead_apple_idx[env->n_dead++] = target_idx;
        // Logging
        env->sum_ticks_apple_collected += (float)env->tick;
        env->agent_returns[a] += 1.0f;
    }
    env->tot_apple_collected += (float)n_collected;
    if (env->shared_rewards){
        for (int i = 0; i < env->num_agents; i++) env->rewards[i] = (float)n_collected;
    }
};

void move_agents(Env* env){
    for (int a = 0; a < env->num_agents; a++){
        const int32_t agent_idx = env->agents_idx[a];
        const int32_t target_idx = env->target[a];
        if (agent_idx != target_idx){
            env->grid[agent_idx] = EMPTY;
            env->grid[target_idx] = AGENT_BASE + a;
            env->agents_idx[a] = target_idx;
        }
    }
};

// ACTION_ZAP.
void fire_beams(Env* env){
    for (int a = 0; a < env->num_agents; a++){
        if ((int32_t)env->actions[a] != ACTION_ZAP) continue;

        env->tot_zaps += 1.0f;
        env->sum_ticks_zap += (float)env->tick; // Logging

        const int32_t agent_idx = env->agents_idx[a];
        const uint8_t agent_dir = env->agents_dir[a];

        for (int k = 0; k < BEAM_MASK_SIZE; k++){
            const int32_t cell_idx = agent_idx + BEAM_DR[k]*env->row_stride[agent_dir] + BEAM_DC[k]*env->col_stride[agent_dir];
            env->beam_idx[env->n_beam++] = cell_idx;
            if (env->grid[cell_idx] >= AGENT_BASE){
                const int32_t victim_id = env->grid[cell_idx] - AGENT_BASE;
                if (env->hit[victim_id]) continue; // A second beam this step changes nothing
                env->hit[victim_id] = 1;
                // Logging
                env->tot_hits += 1.0f;
                env->sum_ticks_hit += (float)env->tick;
            } else if (env->grid[cell_idx] == EMPTY){
                env->grid[cell_idx] = BEAM;
            }
        }
    }
};

// ACTION_CLEAN.
void fire_clean_beams(Env* env){
    for (int a = 0; a < env->num_agents; a++){
        if ((int32_t)env->actions[a] != ACTION_CLEAN) continue;

        env->tot_cleans += 1.0f; // Logging

        const int32_t agent_idx = env->agents_idx[a];
        const uint8_t agent_dir = env->agents_dir[a];

        for (int k = 0; k < BEAM_MASK_SIZE; k++){
            const int32_t cell_idx = agent_idx + BEAM_DR[k]*env->row_stride[agent_dir] + BEAM_DC[k]*env->col_stride[agent_dir];
            env->beam_idx[env->n_beam++] = cell_idx;
            if (env->terrain[cell_idx] == T_DIRT){
                env->terrain[cell_idx] = T_POT_DIRT;
                env->clean_idx[env->n_clean++] = cell_idx;
                env->tot_cells_cleaned += 1.0f; // Logging
            }
            if (env->grid[cell_idx] == EMPTY) env->grid[cell_idx] = CLEAN_BEAM;
        }
    }
};

void c_step(Env* env){
    env->tick += 1;
    memset(env->observations, 0, env->num_agents*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW*sizeof(uint8_t));
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    memset(env->terminals, 0, env->num_agents*sizeof(float));

    apply_respawns(env);
    update_dirt_fraction(env);
    apple_regrow(env);
    spawn_dirt(env);
    beam_clear(env);
    compute_targets(env);
    resolve_conflicts(env);
    collect_apples(env);
    move_agents(env);
    fire_beams(env);
    fire_clean_beams(env);

    // Logging. dirt_fraction is this step's starting value, so a crossing is seen one step late
    env->sum_dirt_fraction += env->dirt_fraction;
    if (env->tick_growth == 0.0f && env->dirt_fraction < THRESHOLD_DEPLETION){
        env->tick_growth = (float)env->tick;
    }

    if (env->tick >= HORIZON){
        add_log(env);
        for (int a = 0; a < env->num_agents; a++) env->terminals[a] = 1.0f;
        c_reset(env);
    }

    compute_observations(env);
};

// ================================================================ render
const Color WALL_COLOR       = (Color){200, 200, 200, 255}; // grey border
const Color EMPTY_COLOR      = (Color){255, 255, 255, 255}; // white
const Color APPLE_COLOR      = (Color){200, 0, 0, 255};     // strong red
const Color BEAM_COLOR       = (Color){255, 255, 153, 255}; // pale yellow
const Color CLEAN_BEAM_COLOR = (Color){173, 216, 230, 255}; // pale blue
const Color RIVER_COLOR      = (Color){0, 0, 128, 255};     // navy, clean water
const Color DIRT_COLOR       = (Color){8, 8, 35, 255};      // almost black

#define HEADER_FONT_SIZE 20
#define HEADER_LINE_HEIGHT 24
#define HEADER_HEIGHT (2*HEADER_LINE_HEIGHT)
#define HEADER_MARGIN 10

struct Client {
    int cell_size;
    int width;   // window width in pixels
    int height;  // window height in pixels
    Texture2D puffers;
};

Client* make_client(Env* env){
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->cell_size = 32;
    client->width  = (env->ascii_width + 2)*client->cell_size;
    client->height = (env->ascii_height + 2)*client->cell_size + HEADER_HEIGHT;
    InitWindow(client->width, client->height, "PufferLib Clean Up");
    SetTargetFPS(60);
    client->puffers = LoadTexture("resources/shared/puffers.png");
    return client;
}

#define NUM_PUFFER_COLORS 8
static const int DIR_TO_SPRITE_STEPS[NUM_DIRS] = {1, 0, 3, 2};

static inline void draw_agent(Texture2D sheet, int x, int y, int size, int agent_id, uint8_t dir){
    int rotation = 90*DIR_TO_SPRITE_STEPS[dir];
    int frame_y = 0;
    if (rotation == 180){
        frame_y = 128;
        rotation = 0;
    }
    int frame_x = 128*(agent_id % NUM_PUFFER_COLORS);
    DrawTexturePro(
        sheet,
        (Rectangle){(float)frame_x, (float)frame_y, 128, 128},
        (Rectangle){x + size/2.0f, y + size/2.0f, (float)size, (float)size},
        (Vector2){size/2.0f, size/2.0f},
        (float)rotation,
        WHITE
    );
}

static inline Color terrain_color(uint8_t cell){
    switch (cell){
        case T_RIVER:    return RIVER_COLOR;
        case T_POT_DIRT: return RIVER_COLOR;
        case T_DIRT:     return DIRT_COLOR;
        default:         return EMPTY_COLOR;
    }
}

// World (0,0) is top-left as reminder :D
void c_render(Env* env){
    if (env->client == NULL){
        env->client = make_client(env);
    }
    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)){
        exit(0);
    }

    BeginDrawing();
    ClearBackground(WALL_COLOR);

    const int n_dirt = N_DIRT_CELLS - env->n_clean;
    char header_buf[64];
    snprintf(header_buf, sizeof(header_buf), "NHits: %d", (int)env->tot_hits);
    DrawText(header_buf, HEADER_MARGIN, 0, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "NCleanHits: %d", (int)env->tot_cells_cleaned);
    DrawText(header_buf, client->width/2 - MeasureText(header_buf, HEADER_FONT_SIZE)/2, 0, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "tick: %d", env->tick);
    DrawText(header_buf, client->width - HEADER_MARGIN - MeasureText(header_buf, HEADER_FONT_SIZE), 0, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "apples collected: %d", (int)env->tot_apple_collected);
    DrawText(header_buf, HEADER_MARGIN, HEADER_LINE_HEIGHT, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "dirty: %d", n_dirt);
    DrawText(header_buf, client->width/2 - MeasureText(header_buf, HEADER_FONT_SIZE)/2, HEADER_LINE_HEIGHT, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "dirt ratio: %.3f", env->dirt_fraction);
    DrawText(header_buf, client->width - HEADER_MARGIN - MeasureText(header_buf, HEADER_FONT_SIZE), HEADER_LINE_HEIGHT, HEADER_FONT_SIZE, BLACK);

    const int cs = client->cell_size;
    const int offset = env->stride*env->pad + env->pad;
    for (int r = 0; r < env->ascii_height; r++){
        const int y = HEADER_HEIGHT + (r + 1)*cs;
        for (int c = 0; c < env->ascii_width; c++){
            const int32_t idx = offset + r*env->stride + c;
            const uint8_t cell = env->grid[idx];
            const int x = (c + 1)*cs;

            // Water underneath, entity on top: the layers never hide each other
            DrawRectangle(x, y, cs, cs, terrain_color(env->terrain[idx]));

            if (cell >= AGENT_BASE){
                int agent_id = cell - AGENT_BASE;
                draw_agent(client->puffers, x, y, cs, agent_id, env->agents_dir[agent_id]);
                continue;
            }

            switch (cell){
                case APPLE:      DrawRectangle(x, y, cs, cs, APPLE_COLOR); break;
                case BEAM:       DrawRectangle(x, y, cs, cs, BEAM_COLOR); break;
                case CLEAN_BEAM: DrawRectangle(x, y, cs, cs, CLEAN_BEAM_COLOR); break;
                default: break;
            }
        }
    }

    EndDrawing();
}

void close_client(Client* client){
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
};

void c_close(Env* env){
    free(env->grid);
    free(env->terrain);
    free(env->apple_idx);
    free(env->init_dirt_idx);
    free(env->init_clean_idx);
    free(env->river_idx);
    free(env->respawn_idx);
    free(env->clean_idx);
    free(env->dead_apple_idx);
    free(env->agents_idx);
    free(env->agents_dir);
    free(env->target);
    free(env->hit);
    free(env->beam_idx);
    free(env->agent_returns);
    if (env->client != NULL){
        close_client(env->client);
    }
};
