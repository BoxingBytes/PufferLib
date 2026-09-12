// Commons Harvest Open -- C port of SocialJax's harvest_open.py
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

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

#define MAX_AGENTS 32
#define HORIZON 1000 // SocialJax's num_inner_steps
// ---------------------------------------------------------------- grid codes
#define EMPTY       0
#define WALL        1   
#define BEAM        2
#define APPLE       3
#define AGENT_BASE  4   // cell holds AGENT_BASE + agent_id

// ------------------------------------------------------------------- actions
#define ACTION_TURN_LEFT     0
#define ACTION_TURN_RIGHT    1
#define ACTION_FORWARD       2
#define ACTION_BACKWARD      3
#define ACTION_STRAFE_LEFT   4
#define ACTION_STRAFE_RIGHT  5
#define ACTION_NOOP          6
#define ACTION_ZAP           7
#define NUM_ACTIONS          8

// ---------------------------------------------------------------- directions
// dir 0 = +row, 1 = +col, 2 = -row, 3 = -col. TURN_LEFT is dir+1, TURN_RIGHT
// is dir-1 (mod 4), matching SocialJax's ROTATIONS table.
#define NUM_DIRS 4
static const int8_t DIR_DROW[NUM_DIRS] = {1, 0, -1, 0};
static const int8_t DIR_DCOL[NUM_DIRS] = {0, 1, 0, -1};

// ------------------------------------------------------------------- regrowth
// Probability that an EMPTY apple cell regrows, by count of live apples in its
// 12-cell mask (the 8 cells of the 3x3 ring plus the 4 orthogonal cells at
// distance 2).
#define REGROW_P_3PLUS 0.025f
#define REGROW_P_2     0.005f
#define REGROW_P_1     0.001f
// count == 0 -> 0.0f, never regrows.

#define APPLE_MASK_SIZE 12
// {dr, dc} for the 3x3 ring (8 cells) plus the 4 orthogonal cells at distance 2.
static const int8_t APPLE_MASK_DR[APPLE_MASK_SIZE] = {-1,-1,-1, 0, 0, 1, 1, 1, -2, 2, 0, 0};
static const int8_t APPLE_MASK_DC[APPLE_MASK_SIZE] = {-1, 0, 1,-1, 1,-1, 0, 1,  0, 0,-2, 2};

// ---------------------------------------------------------------- beams
#define BEAM_MASK_SIZE 4
// The "T" shape in agent heading coordinates.
static const int8_t BEAM_DR[BEAM_MASK_SIZE] = {+1, +1, +1, +2};
static const int8_t BEAM_DC[BEAM_MASK_SIZE] = { 0, -1, +1,  0};  
// ------------------------------------------------------------------ obs shape
#define NUM_OBS_CHANNELS 2
// SocialJax originally one-hot encode into 15 channels. 
// Most of which were useless (inventory/frozen..). For the rest we let the policy one-hot. 
// plane 0 -- terrain/occupancy: EMPTY / WALL / BEAM / APPLE / 4 = an agent
// plane 1 -- agent detail: 0 where no agent, else 1 + 4*id + rel_dir if
//            differentiate_other_agents_in_obs, else 1 + rel_dir.
//            rel_dir = (their_dir - my_dir) & 3.
#define OBS_WINDOW 11

typedef struct Log {
    float time_to_depletion;
    float sustainability;
    float efficiency;
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

    uint8_t* grid;  // (map_height + 2*pad) * stride; border filled with WALL

    // ---------------------------------------------------------- observations
    int32_t start[NUM_DIRS];  // flat relative index of the "behind to the left" corner
    int32_t row_stride[NUM_DIRS];  // flat relative index of the next row in the obs window
    int32_t col_stride[NUM_DIRS];  // flat relative index of the next col in the obs window

    // ------------------------------------------------------------ config
    bool beam_blocks_movement;    // default true
    bool differentiate_other_agents_in_obs;  // default true
    bool shared_rewards;          // default false

    // ------------------------------------------------------------ agents
    int32_t *agents_idx;  // padded flat cell index
    uint8_t *agents_dir;  // 0..3

    // ---------------------------------------------------- step scratch
    int32_t *target;  // proposed then resolved destination cell
    uint8_t *hit;     // beamed this step, consumed by pick_respawn_cells next step. Stores agent_ids TODO carefull about an agent being beamed twice
    int n_hit;

    // ------------------------------------------------- static cell sets
    // Parsed once from the ASCII map (see init). Padded flat indices.
    int32_t *apple_idx;      // 'A' -- 64 cells on the open map
    int n_apple;
    int32_t apple_mask_offset[APPLE_MASK_SIZE]; // flat index deltas for the 12-cell regrow mask
    int32_t *respawn_idx;    // 'P' -- 60 cells; the only cells respawn uses
    int n_respawn;
    int32_t *respawn_in_idx; // 'Q' -- 2 cells; used only at reset
    int n_respawn_in;

    int32_t *new_apple;      // grid_idx, (n_apple,) see apple_regrow
    int n_apple_new;
    int n_apple_alive;

    // -------------------------------------------------------------- beams
    int32_t *beam_idx;   // Write beamed cells. For faster clearing lookup only
    int n_beam;

    // ---------------------------------------------------------------- rng
    uint32_t rng;  // Because rand() gave issues with low probs

    int tick;

    // -------------------------------------------------------------- map 
    const char *map_ASCII[16];

    // -------------------------------------------------------------- logging
    float sum_ticks_apple_collected; // For computing sustainability
    float tot_apple_collected; // For computing sustainability
    float tot_zaps;
    float tot_hits;
    float sum_ticks_zap; // For computing zap_timing
    float sum_ticks_hit; // For computing hit_timing
    float *agent_returns; // For computing equality
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

// Reseeds. Substitutes 1 for a zero seed. Exposed so eval rollouts can share
// an apple-regrowth sequence across a Schelling-diagram mixture sweep.
void c_seed(Env* env, uint32_t seed){
    env->rng = (seed == 0) ? 1 : seed;
}

// ================================================================== init
// Parses an ASCII map (maps.h) into grid geometry and the static cell sets,
// allocates everything sized by the map, fills the padding border with WALL,
// and precomputes obs_off. Asserts pad >= 2 and n_respawn >= num_agents.
// Reads: map string. Writes: all of Env except the PufferLib buffers.
void init(Env* env){
    const char *map[16] = {
        "AAA    A      A    AAA",
        "AA    AAA    AAA    AA",
        "A    AAAAA  AAAAA    A",
        "      AAA    AAA      ",
        "       A      A       ",
        "  A                A  ",
        " AAA  Q        Q  AAA ",
        "AAAAA            AAAAA",
        " AAA              AAA ",
        "  A                A  ",
        "                      ",
        "                      ",
        "                      ",
        "  PPPPPPPPPPPPPPPPPP  ",
        " PPPPPPPPPPPPPPPPPPPP ",
        "PPPPPPPPPPPPPPPPPPPPPP",
    };
    memcpy(env->map_ASCII, map, sizeof(map));

    env->log = (Log){0};

    env->ascii_width = strlen(map[0]);
    env->ascii_height = sizeof(map) / sizeof(map[0]);
    env->pad = OBS_WINDOW - 1; //Because obs start one cell behind the agent
    env->stride = env->ascii_width + 2*env->pad;

    // Observation windows sweep from "left-to-right" relative to agent dir, from behind to ahead.
    for (int d = 0; d < NUM_DIRS; d++){
        env->row_stride[d] = DIR_DROW[d]*env->stride + DIR_DCOL[d];
        env->col_stride[d] = DIR_DCOL[d]*env->stride - DIR_DROW[d];
        env->start[d] = -env->row_stride[d] - (OBS_WINDOW/2)*env->col_stride[d];
    }

    // Makes it easier to compute the 12-cell mask for apple regrowth.
    for (int k = 0; k < APPLE_MASK_SIZE; k++){
        env->apple_mask_offset[k] = APPLE_MASK_DR[k]*env->stride + APPLE_MASK_DC[k];
    }

    const int tot_flattened = (env->ascii_height + 2*env->pad)*(env->stride);
    env->grid = (uint8_t*)calloc(tot_flattened, sizeof(uint8_t));
    memset(env->grid, WALL, tot_flattened*sizeof(uint8_t));

    c_seed(env, 0x12345678);

    env->apple_idx = (int32_t*)calloc(64, sizeof(int32_t));
    env->n_apple = 0;
    env->respawn_idx = (int32_t*)calloc(60, sizeof(int32_t));
    env->n_respawn = 0;
    env->respawn_in_idx = (int32_t*)calloc(2, sizeof(int32_t));
    env->n_respawn_in = 0;
    env->new_apple = (int32_t*)calloc(64, sizeof(int32_t));

    env->agents_idx = (int32_t*)calloc(env->num_agents, sizeof(int32_t));
    env->agents_dir = (uint8_t*)calloc(env->num_agents, sizeof(uint8_t));
    env->target = (int32_t*)calloc(env->num_agents, sizeof(int32_t));
    env->hit = (uint8_t*)calloc(env->num_agents, sizeof(uint8_t));
    env->n_hit = 0;
    env->beam_idx = (int32_t*)calloc(4*env->num_agents, sizeof(int32_t));
    env->n_beam = 0;

    const int offset = env->stride*env->pad + env->pad;

    for (int r = 0; r < env->ascii_height; r++){
        for (int c = 0; c < env->ascii_width; c++){
            const char cell = map[r][c];
            const int idx = offset + r*env->stride + c;
            env->grid[idx] = EMPTY;
            switch (cell){
                case 'A':
                    env->apple_idx[env->n_apple++] = idx;
                    break;
                case 'P':
                    env->respawn_idx[env->n_respawn++] = idx;
                    break;
                case 'Q':
                    env->respawn_in_idx[env->n_respawn_in++] = idx;
                    break;
                default:
                    break;
            }
        }
    };

    // Logging
    env->agent_returns = (float*)calloc(env->num_agents, sizeof(float));
};


// OBS_WINDOW x OBS_WINDOW window per agent, NUM_OBS_CHANNELS channels.
// Each agent's window is oriented relative to its heading, with the agent at
// the center of the window's bottom 2nd row. 
void compute_observations(Env* env){

    // Channel 0 (terrain/occupancy): branch-free pass over every window cell.
    // Any agent-occupied cell collapses to the constant AGENT_BASE (which
    // agent doesn't matter for this channel).
    for (int a = 0; a < env->num_agents; a++){
        const int32_t channel0_base = a*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW;
        const int32_t agent_idx = env->agents_idx[a];
        const uint8_t agent_dir = env->agents_dir[a];
        // Locate "behind to the left" corner.
        const int32_t start_idx = agent_idx + env->start[agent_dir];

        for (int r = 0; r < OBS_WINDOW; r++){
            const int32_t row_idx = start_idx + r*env->row_stride[agent_dir];
            for (int c = 0; c < OBS_WINDOW; c++){
                const int32_t cell_idx = row_idx + c*env->col_stride[agent_dir];
                const uint8_t cell_content = env->grid[cell_idx];
                env->observations[channel0_base + r*OBS_WINDOW + c] =
                    (cell_content < AGENT_BASE) ? cell_content : AGENT_BASE;
            }
        }
    }

    // Channel 1 (agent detail): O(num_agents^2) instead of O(num_agents*window^2).
    // Relies on c_step's per-tick memset of env->observations for the 0 default,
    // and only visits actual agent pairs instead of every window cell.
    for (int a = 0; a < env->num_agents; a++){
        const int32_t channel1_base = a*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW + OBS_WINDOW*OBS_WINDOW;
        const uint8_t agent_dir = env->agents_dir[a];
        const int32_t agent_row = env->agents_idx[a] / env->stride;
        const int32_t agent_col = env->agents_idx[a] % env->stride;

        // b == a included on purpose: matches the old loop's behavior of
        // marking an agent's own cell in its own channel 1.
        for (int b = 0; b < env->num_agents; b++){
            const int32_t other_row = env->agents_idx[b] / env->stride;
            const int32_t other_col = env->agents_idx[b] % env->stride;
            const int32_t drow = other_row - agent_row;
            const int32_t dcol = other_col - agent_col;

            // Rotate (drow, dcol) into agent a's forward/right frame.
            int32_t fwd, right;
            switch (agent_dir){
                case 0: fwd =  drow; right = -dcol; break;
                case 1: fwd =  dcol; right =  drow; break;
                case 2: fwd = -drow; right =  dcol; break;
                default: fwd = -dcol; right = -drow; break; // case 3
            }

            const int32_t r = fwd + 1;
            const int32_t c = right + OBS_WINDOW/2;
            if (r < 0 || r >= OBS_WINDOW || c < 0 || c >= OBS_WINDOW) continue;

            const uint8_t other_dir = env->agents_dir[b];
            const int32_t rel_dir = (other_dir - agent_dir) & 3; // works because modulo is 2^n and rel_dir can be < 0
            const int32_t k = r*OBS_WINDOW + c;

            if (env->differentiate_other_agents_in_obs){
                // Encoding both agent_id & direction in one go, bad idea?
                env->observations[channel1_base + k] = 1 + 4*b + rel_dir;
            } else {
                env->observations[channel1_base + k] = 1 + rel_dir;
            }
        }
    }
};


// Matches SocialJax: agents 0 and 1 seat on the 2 'Q' cells, the rest on 'P'
void c_reset(Env* env){
    env->tick = 0;
    memset(env->hit, 0, env->num_agents*sizeof(uint8_t));
    env->n_hit = 0;
    memset(env->target, 0, env->num_agents*sizeof(int32_t));
    memset(env->beam_idx, 0, 4*env->num_agents*sizeof(int32_t));
    env->n_beam = 0;
    memset(env->new_apple, 0, env->n_apple*sizeof(int32_t));

    // Puffer
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    memset(env->terminals, 0, env->num_agents*sizeof(float));
    memset(env->observations, 0, env->num_agents*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW*sizeof(uint8_t));

    // Rebuild the grid
    const int offset = env->stride*env->pad + env->pad;
    for (int r = 0; r < env->ascii_height; r++){
        memset(env->grid + offset + r*env->stride, EMPTY, env->ascii_width*sizeof(uint8_t));
    }

    // All apple cells live
    for (int i = 0; i < env->n_apple; i++){
        env->grid[env->apple_idx[i]] = APPLE;
    }

    // Agents 0/1 seat on 'Q' cells, the rest on 'P' cells
    for (int i = 0; i < env->num_agents; i++){
        int32_t idx;
        if (i < env->n_respawn_in){
            idx = env->respawn_in_idx[i];
        } else {
            do {
                idx = env->respawn_idx[rnd(env) % env->n_respawn];
            } while (env->grid[idx] != EMPTY);
        }
        env->grid[idx] = AGENT_BASE + i;
        env->agents_idx[i] = idx;
        env->agents_dir[i] = rnd(env) % NUM_DIRS;
    }

    // Logging
    env->n_apple_alive = env->n_apple;
    env->sum_ticks_apple_collected = 0.0f;
    env->tot_apple_collected = 0.0f;
    env->tot_zaps = 0.0f;
    env->tot_hits = 0.0f;
    env->sum_ticks_zap = 0.0f;
    env->sum_ticks_hit = 0.0f;
    memset(env->agent_returns, 0, env->num_agents*sizeof(float));

    compute_observations(env);
};

// Uses Gini coefficient as in inequity aversion (Hughes et. al 2018)
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
    if (env->n_apple_alive > 0) env->log.time_to_depletion += (float)HORIZON;

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
    env->log.zap_rate += env->tot_zaps / (float)HORIZON;
    env->log.hit_rate += env->tot_hits / (float)HORIZON;

    env->log.perf += env->tot_apple_collected/(env->num_agents*HORIZON); // Loose
    env->log.score += env->tot_apple_collected;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->tot_apple_collected;
    
    env->log.n++;
};

void apply_respawns(Env* env){
    for (int i = 0; i < env->n_hit; i++){
        const int32_t agent_id = env->hit[i];
        int32_t respawn_cell;
        do {
            respawn_cell = rnd(env) % env->n_respawn;
        } while (env->grid[env->respawn_idx[respawn_cell]] != EMPTY);
        uint8_t new_dir = (rnd(env) % (NUM_DIRS - 1)) + 1; // 1..3, never 0 = original SocialJax
        env->grid[env->agents_idx[agent_id]] = EMPTY;
        env->agents_idx[agent_id] = env->respawn_idx[respawn_cell];
        env->agents_dir[agent_id] = new_dir;
        env->grid[env->agents_idx[agent_id]] = AGENT_BASE + agent_id;
    }
    env->n_hit = 0;
};

void apple_regrow(Env* env){

    env->n_apple_new = 0;
    // Compute regrowth for each apple cell. 
    for (int i = 0; i < env->n_apple; i++){
        int32_t apple_cell = env->apple_idx[i];
        
        // We only look to regrow apples where it's empty 
        if (env->grid[apple_cell] != EMPTY) continue;

        int live_count = 0;
        for (int k = 0; k < APPLE_MASK_SIZE; k++){
            int32_t neighbor_cell = apple_cell + env->apple_mask_offset[k];
            if (env->grid[neighbor_cell] == APPLE) live_count++;
        }

        float p = 0.0f;
        if (live_count == 1) p = REGROW_P_1;
        else if (live_count == 2) p = REGROW_P_2;
        else if (live_count >= 3) p = REGROW_P_3PLUS;
        
        if (rndf(env) < p) env->new_apple[env->n_apple_new++] = apple_cell;            
    }

    // Apply the regrowth
    for (int i = 0; i < env->n_apple_new; i++){
        env->n_apple_alive++;
        int32_t apple_cell = env->new_apple[i];
        env->grid[apple_cell] = APPLE;
    }
};

void beam_clear(Env* env){
    for (int i = 0; i < env->n_beam; i++){
        int32_t cell = env->beam_idx[i];
        if (env->grid[cell] == BEAM) env->grid[cell] = EMPTY;
    }
    env->n_beam = 0;
};

// Compute next_cell targets. Turns take effect immediately. Moves are
// egocentric w.r.t. agent heading -- FORWARD is always forward w.r.t. heading.
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
                break;
            default:
                printf("compute_targets: invalid action %d\n", action);
                break;
        }

        if (env->grid[target_idx] == WALL || (env->grid[target_idx] == BEAM)){
            target_idx = agent_idx;
        }

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

// Rewards agents whose RESOLVED destination holds an apple. Must run before
// move_agents stamps over it. 
void collect_apples(Env* env){
    for (int a = 0; a < env->num_agents; a++){
        const int32_t target_idx = env->target[a];
        if (env->grid[target_idx] == APPLE){
            env->rewards[a] += 1.0f;
            env->grid[target_idx] = EMPTY;
            // Logging
            env->sum_ticks_apple_collected += (float)env->tick;
            env->tot_apple_collected += 1.0f;
            env->n_apple_alive--;
            env->agent_returns[a] += 1.0f;
        }
    }
};

// Applies resolved destinations to the grid.
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
                env->hit[env->n_hit++] = victim_id;
                env->sum_ticks_hit += (float)env->tick; // Logging
            } else if (env->grid[cell_idx] == EMPTY){
                env->grid[cell_idx] = BEAM;
            }
        }
    }
    env->tot_hits += (float)env->n_hit;
};

void c_step(Env* env){
    env->tick += 1;
    memset(env->observations, 0, env->num_agents*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW*sizeof(uint8_t));
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    memset(env->terminals, 0, env->num_agents*sizeof(float));

    apply_respawns(env);
    apple_regrow(env);
    if (!env->beam_blocks_movement) beam_clear(env);
    compute_targets(env);
    resolve_conflicts(env);
    collect_apples(env);
    move_agents(env);
    if (env->beam_blocks_movement) beam_clear(env);
    fire_beams(env);

    // Logging
    if (env->n_apple_alive == 0 && env->log.time_to_depletion == 0) env->log.time_to_depletion = (float)env->tick;

    if (env->tick >= HORIZON){
        add_log(env);
        for (int a = 0; a < env->num_agents; a++) env->terminals[a] = 1.0f;
        c_reset(env);
    }

    compute_observations(env);
};

// ================================================================ render
const Color WALL_COLOR  = (Color){173, 216, 230, 255}; // pale blue
const Color EMPTY_COLOR = (Color){255, 255, 255, 255}; // white
const Color APPLE_COLOR = (Color){200, 0, 0, 255};      // strong red
const Color BEAM_COLOR  = (Color){255, 255, 153, 255};  // pale yellow

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
    InitWindow(client->width, client->height, "PufferLib Commons Harvest");
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

    char header_buf[64];
    snprintf(header_buf, sizeof(header_buf), "NZap: %d", (int)env->tot_zaps);
    DrawText(header_buf, HEADER_MARGIN, 0, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "NHits: %d", (int)env->tot_hits);
    DrawText(header_buf, client->width/2 - MeasureText(header_buf, HEADER_FONT_SIZE)/2, 0, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "tick: %d", env->tick);
    DrawText(header_buf, client->width - HEADER_MARGIN - MeasureText(header_buf, HEADER_FONT_SIZE), 0, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "apples collected: %d", (int)env->tot_apple_collected);
    DrawText(header_buf, HEADER_MARGIN, HEADER_LINE_HEIGHT, HEADER_FONT_SIZE, BLACK);

    snprintf(header_buf, sizeof(header_buf), "equality: %.3f", compute_equality(env));
    DrawText(header_buf, client->width/2 - MeasureText(header_buf, HEADER_FONT_SIZE)/2, HEADER_LINE_HEIGHT, HEADER_FONT_SIZE, BLACK);

    const int cs = client->cell_size;
    const int offset = env->stride*env->pad + env->pad;
    for (int r = 0; r < env->ascii_height; r++){
        const int y = HEADER_HEIGHT + (r + 1)*cs;
        for (int c = 0; c < env->ascii_width; c++){
            const uint8_t cell = env->grid[offset + r*env->stride + c];
            const int x = (c + 1)*cs;

            if (cell >= AGENT_BASE){
                int agent_id = cell - AGENT_BASE;
                DrawRectangle(x, y, cs, cs, EMPTY_COLOR);
                draw_agent(client->puffers, x, y, cs, agent_id, env->agents_dir[agent_id]);
                continue;
            }

            Color color;
            switch (cell){
                case APPLE: color = APPLE_COLOR; break;
                case BEAM:  color = BEAM_COLOR;  break;
                default:    color = EMPTY_COLOR; break;
            }
            DrawRectangle(x, y, cs, cs, color);
        }
    }

    EndDrawing();
}

void close_client(Client* client){
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}

void c_close(Env* env){
    free(env->grid);
    free(env->apple_idx);
    free(env->respawn_idx);
    free(env->respawn_in_idx);
    free(env->new_apple);
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
