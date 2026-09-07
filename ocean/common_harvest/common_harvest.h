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
#include "raylib.h"

#define MAX_AGENTS 32

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

// ------------------------------------------------------------------- regrowth
// Probability that an EMPTY apple cell regrows, by count of live apples in its
// 12-cell mask (the 8 cells of the 3x3 ring plus the 4 orthogonal cells at
// distance 2).
#define REGROW_P_3PLUS 0.025f
#define REGROW_P_2     0.005f
#define REGROW_P_1     0.001f
// count == 0 -> 0.0f, never regrows.

// ------------------------------------------------------------------ obs shape
#define NUM_OBS_PLANES 2
// plane 0 -- terrain/occupancy: EMPTY / WALL / BEAM / APPLE / 4 = an agent
// plane 1 -- agent detail: 0 where no agent, else 1 + 4*id + rel_dir if
//            differentiate_other_agents_in_obs, else 1 + rel_dir.
//            rel_dir = (their_dir - my_dir) & 3.
// TODO: a way to get rid of the second plane? Mostly useless
#define OBS_PLANE_AGENT 4

typedef struct Log {
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
    uint8_t* observations;  // (num_agents, NUM_OBS_PLANES, OBS_SIZE, OBS_SIZE)
    int32_t* actions;       // (num_agents,) 
    float* rewards;         // (num_agents,)
    unsigned char* terminals;  // (num_agents,)
    unsigned char *masks;  // (num_agents,) 1 = agent is alive, 0 = dead
    int num_agents;
    Log log;
    Client* client;

    // ---------------------------------------------------------- geometry
    // All stored indices are PADDED flat indices; only the renderer converts back.
    int ascii_height;    // map_height, UNPADDED, from ASCII map
    int ascii_width;     // map_width, UNPADDED, from ASCII map
    int pad;       // OBS_SIZE - 1 
    int stride;    // map_width + 2*pad
    int obs_size;   

    uint8_t* grid;  // (map_height + 2*pad) * stride; border filled with WALL

    // ------------------------------------------------------------ config
    int horizon;                  // SocialJax num_inner_steps, default 1000
    bool beam_blocks_movement;    // default true
    bool differentiate_other_agents_in_obs;  // default true
    bool shared_rewards;          // default false
    // TODO (chose): relative vs absolute agent id in obs plane 1. Relative
    // ((other - me + N) % N) makes the observation permutation-consistent,
    // which matters iff training with shared policy weights.
    bool relative_agent_ids;

    // ------------------------------------------------------------ agents
    int32_t *agents_idx;  // padded flat cell index
    uint8_t *agents_dir;  // 0..3

    // ---------------------------------------------------- step scratch
    int32_t *target;  // proposed then resolved destination cell
    uint8_t *hit;     // beamed this step, consumed by pick_respawn_cells next step TODO carefull about an agent being beamed twice
    int n_hit;

    // ------------------------------------------------- static cell sets
    // Parsed once from the ASCII map (see init). Padded flat indices.
    int32_t *apple_idx;      // 'A' -- 64 cells on the open map
    int n_apple;
    int32_t *respawn_idx;    // 'P' -- 60 cells; the only cells respawn uses
    int n_respawn;
    int32_t *respawn_in_idx; // 'Q' -- 2 cells; used only at reset
    int n_respawn_in;

    uint8_t *new_apple;      // (n_apple,) regrow double-buffer, see apple_regrow

    // -------------------------------------------------------------- beams
    int32_t *beam_idx;   // Write beamed cells. For faster clearing lookup only
    int n_beam;

    // ------------------------------------------------- obs offset tables
    // enables to rotate and translate obs based on agent heading
    int32_t *obs_off[NUM_DIRS];
    uint8_t xlat[NUM_OBS_PLANES][256];

    // ---------------------------------------------------------------- rng
    uint32_t rng;  // Because rand() gave issues with low probs

    int tick;

    // -------------------------------------------------------------- map 
    const char *map_ASCII[16];
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
    env->pad = env->obs_size - 1; //Because obs start one cell behind the agent
    env->stride = env->ascii_width + 2*env->pad;

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
    env->new_apple = (uint8_t*)calloc(64, sizeof(uint8_t));

    env->agents_idx = (int32_t*)calloc(env->num_agents, sizeof(int32_t));
    env->agents_dir = (uint8_t*)calloc(env->num_agents, sizeof(uint8_t));
    env->target = (int32_t*)calloc(env->num_agents, sizeof(int32_t));
    env->hit = (uint8_t*)calloc(env->num_agents, sizeof(uint8_t));
    env->n_hit = 0;
    env->beam_idx = (int32_t*)calloc(4*env->num_agents, sizeof(int32_t));
    env->n_beam = 0;

    for (int d = 0; d < NUM_DIRS; d++){
        env->obs_off[d] = (int32_t*)calloc(env->obs_size*env->obs_size, sizeof(int32_t));
    }

    const int offset = env->stride + env->pad;

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
};


// Egocentric OBS_SIZE x OBS_SIZE window per agent, NUM_OBS_PLANES planes, agent
// at (1, OBS_SIZE/2) facing toward increasing row: 1 cell behind, OBS_SIZE-2
// ahead. Emits integer codes; one-hot/embedding is the policy's job.
// obs[k] = xlat[plane][grid[agents_idx[i] + obs_off[dir][k]]], with xlat
// rebuilt per observer so that other agents' headings come out relative
// ((their_dir - my_dir) & 3). Runs after fire_beams so beams are visible.
// This loop is the whole step's cost: num_agents * obs_size^2 * NUM_OBS_PLANES
// gathers and stores, against roughly 25 grid writes for everything else.
// Reads: grid, agents_idx, agents_dir. Writes: xlat, observations.
void compute_observations(Env* env);


// ================================================================= api
// Full episode reset: rebuilds the grid from the static cell sets (all apple
// cells live), seats agents, clears respawn_target/beams/tick, and computes
// observations.
// Matches SocialJax: agents 0 and 1 seat on the 2 'Q' cells, the rest on 'P'
// cells.
void c_reset(Env* env){
    env->tick = 0;
    memset(env->hit, 0, env->num_agents*sizeof(uint8_t));
    env->n_hit = 0;
    memset(env->target, 0, env->num_agents*sizeof(int32_t));
    memset(env->beam_idx, 0, 4*env->num_agents*sizeof(int32_t));
    env->n_beam = 0;
    memset(env->new_apple, 0, env->n_apple*sizeof(uint8_t));

    // Puffer
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    memset(env->terminals, 0, env->num_agents*sizeof(unsigned char));
    memset(env->observations, 0, env->num_agents*NUM_OBS_PLANES*env->obs_size*env->obs_size*sizeof(uint8_t));
    memset(env->masks, 1, env->num_agents*sizeof(unsigned char));
    env->log = (Log){0};

    // Rebuild the grid
    const int offset = env->stride + env->pad;
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
};


void add_log(Env* env);

// ============================================================ step passes
// Called in this order by c_step. Each is a side-effecting pass over Env; the
// reads/writes below are the complete set.

// Relocates agents killed last step onto their pre-chosen respawn cell and
// gives them a fresh random heading.
// MUST clear all vacated cells before stamping any new one: a reborn agent's
// target may be another reborn agent's death cell, and a single fused loop
// would erase one of them depending only on agent index order.
// OPEN: heading draw range. SocialJax uses maxval=3, so agents never spawn
// facing direction 3; use 4.
// Reads: respawn_target, agents_idx. Writes: grid, agents_idx, agents_dir,
// respawn_target.
void apply_respawns(Env* env);

// Regrows apples with probability by live-apple count in the 12-cell mask:
// >=3 -> 0.025, 2 -> 0.005, 1 -> 0.001, 0 -> never. Cells already holding an
// APPLE are left alone; cells holding anything else (agent, corpse, beam mark)
// are skipped entirely, so no apple ever grows under an agent and a beamed cell
// loses one step of regrow opportunity.
// Results go into new_apple[] and are applied in a second pass, so regrowth is
// simultaneous: a cell that grows here must not count as a neighbour for a
// later cell in the same pass. A 64-byte buffer, not a grid copy.
// Reads: apple_idx, grid, rng. Writes: new_apple, grid (only cells that grow).
void apple_regrow(Env* env);

// Clears this env's beam marks. Walks beam_idx rather than scanning the grid,
// and MUST guard each write with grid[c] == BEAM: a recorded cell may since
// have been overwritten by an agent stamp or a regrown apple, and an unguarded
// clear would delete them.
// Reads: beam_idx, grid. Writes: grid, n_beam.
void beam_clear(Env* env);

// Turns take effect immediately (visible in this step's observation). Moves are
// egocentric w.r.t. agent heading -- FORWARD is always forward w.r.t. heading,
// not "up" in the world. Turn and zap actions target the agent's own cell.
// Terrain blocking happens here, not in resolve_conflicts: a target holding
// WALL, or holding BEAM while beam_blocks_movement, collapses to the agent's
// own cell. Padding means no clipping is needed.
// Reads: actions, agents_idx, agents_dir, grid. Writes: target, agents_dir.
void compute_targets(Env* env);

// Agent-agent conflict resolution, Melting Pot semantics (see
// socialjax/environments/movement.py): a non-mover always keeps its cell;
// among movers contesting one cell a uniformly random winner is chosen and the
// rest revert; 2-cycles (swaps) are blocked; cycles of length >= 3 are allowed;
// "trains" (B enters the cell A vacates in the same step) are allowed, and a
// revert cascades to whoever was following, iterated to a fixed point.
// Postcondition: all N destinations are pairwise distinct.
// Touches NO grid -- pure position arithmetic. Terrain was handled in
// compute_targets; a grid read here means something has leaked.
// Reads: target, agents_idx, rng. Writes: target.
void resolve_conflicts(Env* env);

// Rewards agents whose RESOLVED destination holds an apple. Must run before
// move_agents stamps over it. A non-mover never collects (its own cell holds
// its own agent code, and no apple can grow beneath it).
// Reads: target, grid. Writes: rewards, log.
void collect_apples(Env* env);

// Applies resolved destinations to the grid. Same hazard as apply_respawns:
// all clears before all stamps, or a train erases the agent it is following.
// Reads: target, agents_idx. Writes: grid, agents_idx.
void move_agents(Env* env);

// Fires the reverse-T beam for every agent whose action is ZAP: the 3-wide
// line at distance 1 (forward, forward+left, forward+right) plus the stem at
// distance 2 forward. No line of sight, no stopping at the first hit, all four
// cells always resolved, no cooldown.
// A cell holding an agent code is a hit -- one hit is lethal, and the victim
// may be a corpse or may be simultaneously killing its killer. A BEAM mark is
// written only to EMPTY cells, so a beam that lands on an agent, apple or wall
// leaves no visible trace: nobody but the victim can see that a hit occurred.
// All four target cells are recorded in beam_idx regardless.
// Reads: actions, agents_idx, agents_dir, grid. Writes: grid, beam_idx,
// n_beam, hit, log.
void fire_beams(Env* env);

// Chooses each hit agent's respawn cell, consumed by apply_respawns at the top
// of the next step (this is what makes death deferred). Draws from respawn_idx
// ('P') only, never 'Q'. A cell is available if no SURVIVOR occupies it -- a
// hit agent standing on a 'P' cell vacates it -- and no two agents reborn in
// the same step may pick the same cell.
// Reads: hit, agents_idx, grid, respawn_idx, rng. Writes: respawn_target.
void pick_respawn_cells(Env* env);

// Runs the passes above in order, then rewards/terminals/tick. On tick ==
// horizon it resets in place and returns the fresh observation, with the
// horizon step's reward left intact.
void c_step(Env* env);

// ================================================================ render
const Color WALL_COLOR  = (Color){173, 216, 230, 255}; // pale blue
const Color EMPTY_COLOR = (Color){255, 255, 255, 255}; // white
const Color APPLE_COLOR = (Color){200, 0, 0, 255};      // strong red
const Color BEAM_COLOR  = (Color){255, 255, 153, 255};  // pale yellow

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
    client->height = (env->ascii_height + 2)*client->cell_size;
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

    const int cs = client->cell_size;
    const int offset = env->stride + env->pad;
    for (int r = 0; r < env->ascii_height; r++){
        const int y = (r + 1)*cs;
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
    for (int d = 0; d < NUM_DIRS; d++){
        free(env->obs_off[d]);
    }
    if (env->client != NULL){
        close_client(env->client);
    }
};
