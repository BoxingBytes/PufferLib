#include <time.h>
#include "common_harvest.h"


void print_observation(Env* env, int agent_id) {
    if (agent_id < 0 || agent_id >= env->num_agents) {
        printf("print_observation: invalid agent_id %d (num_agents=%d)\n", agent_id, env->num_agents);
        return;
    }

    const int size = OBS_WINDOW;
    const int32_t base = agent_id*NUM_OBS_CHANNELS*size*size;
    const uint8_t* channel0 = env->observations + base;
    const uint8_t* channel1 = env->observations + base + size*size;
    const int agent_row = 1;
    const int agent_col = size/2;

    printf("=== agent %d obs (dir=%d) -- channel 0 (terrain) ===\n", agent_id, env->agents_dir[agent_id]);
    for (int r = size - 1; r >= 0; r--) {
        for (int c = 0; c < size; c++) {
            char buf[16];
            uint8_t v = channel0[r*size + c];
            if (r == agent_row && c == agent_col) snprintf(buf, sizeof(buf), "[%d]", v);
            else snprintf(buf, sizeof(buf), "%d", v);
            printf("%6s", buf);
        }
        printf("\n");
    }

    printf("--- channel 1 (agent detail) ---\n");
    for (int r = size - 1; r >= 0; r--) {
        for (int c = 0; c < size; c++) {
            char buf[16];
            uint8_t v = channel1[r*size + c];
            if (r == agent_row && c == agent_col) snprintf(buf, sizeof(buf), "[%d]", v);
            else snprintf(buf, sizeof(buf), "%d", v);
            printf("%6s", buf);
        }
        printf("\n");
    }
    printf("\n");
}

void print_first_last_observations(Env* env) {
    print_observation(env, 0);
    print_observation(env, env->num_agents - 1);
}

#define PERF_ACTION_BUFFER_TICKS 1024

void performance_test() {
    const long test_time = 30;
    Env env = {
        .num_agents = 4,
        .differentiate_other_agents_in_obs = false,
        .beam_blocks_movement = true,
        .shared_rewards = false,
        .rng = 42,
    };
    env.observations = (uint8_t*)calloc(env.num_agents*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW, sizeof(uint8_t));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (float*)calloc(env.num_agents, sizeof(float));
    env.actions = (float*)calloc(env.num_agents, sizeof(float));

    init(&env);
    c_reset(&env);

    int32_t action_buffer[PERF_ACTION_BUFFER_TICKS][env.num_agents];
    for (int t = 0; t < PERF_ACTION_BUFFER_TICKS; t++){
        for (int a = 0; a < env.num_agents; a++){
            action_buffer[t][a] = rand() % NUM_ACTIONS;
        }
    }

    long start = time(NULL);
    long steps = 0;
    while (time(NULL) - start < test_time) {
        int32_t* tick_actions = action_buffer[steps % PERF_ACTION_BUFFER_TICKS];
        for (int a = 0; a < env.num_agents; a++){
            env.actions[a] = tick_actions[a];
        }
        c_step(&env);
        steps++;
    }
    long elapsed = time(NULL) - start;

    printf("SPS: %ld\n", steps/elapsed);
    printf("SPS per agent: %ld\n", (steps*env.num_agents)/elapsed);

    c_close(&env);
    free(env.observations);
    free(env.rewards);
    free(env.terminals);
    free(env.actions);
}

void demo() {
    Env env = {
        .num_agents = 7,
        .differentiate_other_agents_in_obs = true,
        .beam_blocks_movement = true,
        .shared_rewards = true,
        .rng = 42,
    };
    env.observations = (uint8_t*)calloc(env.num_agents*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW, sizeof(uint8_t));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (float*)calloc(env.num_agents, sizeof(float));
    env.actions = (float*)calloc(env.num_agents, sizeof(float));

    init(&env);
    c_reset(&env);
    c_render(&env);

    const int human_agent_idx = 0;
    while (!WindowShouldClose()) {
        for (int a = 0; a < env.num_agents; a++){
            env.actions[a] = rand() % NUM_ACTIONS;
        }

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            int32_t action = ACTION_NOOP;
            if (IsKeyDown(KEY_UP))         action = ACTION_FORWARD;
            else if (IsKeyDown(KEY_DOWN))  action = ACTION_BACKWARD;
            else if (IsKeyDown(KEY_LEFT))  action = ACTION_STRAFE_LEFT;
            else if (IsKeyDown(KEY_RIGHT)) action = ACTION_STRAFE_RIGHT;
            else if (IsKeyDown(KEY_Q))     action = ACTION_TURN_LEFT;
            else if (IsKeyDown(KEY_E))     action = ACTION_TURN_RIGHT;
            else if (IsKeyDown(KEY_SPACE)) action = ACTION_ZAP;
            env.actions[human_agent_idx] = action;
        }

        c_step(&env);

        float sum = 0.0f;
        for (int a = 0; a < env.num_agents; a++) sum += env.rewards[a];
        if (sum != 0.0f){
            printf("t=%4d r:", env.tick);
            for (int a = 0; a < env.num_agents; a++) printf(" %6.3f", env.rewards[a]);
            printf(" | sum=%.3f apples=%d\n", sum, (int)env.tot_apple_collected);
        }

        c_render(&env);
    }
    c_close(&env);

    free(env.observations);
    free(env.rewards);
    free(env.terminals);
    free(env.actions);
}

// Every step, sum_a rewards[a] must equal the apples collected that step, in
// both reward modes. Under shared_rewards all agents must also get equal shares.
void test_rewards(bool shared_rewards) {
    Env env = {
        .num_agents = 7,
        .differentiate_other_agents_in_obs = true,
        .beam_blocks_movement = true,
        .shared_rewards = shared_rewards,
        .rng = 42,
    };
    env.observations = (uint8_t*)calloc(env.num_agents*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW, sizeof(uint8_t));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (float*)calloc(env.num_agents, sizeof(float));
    env.actions = (float*)calloc(env.num_agents, sizeof(float));

    init(&env);
    c_reset(&env);

    const int num_episodes = 5;
    int failures = 0;
    float total_reward = 0.0f;
    float prev_apples = 0.0f;

    for (int step = 0; step < num_episodes*HORIZON; step++){
        for (int a = 0; a < env.num_agents; a++) env.actions[a] = rand() % NUM_ACTIONS;
        bool last = (env.tick == HORIZON - 1);
        c_step(&env);

        float sum = 0.0f;
        for (int a = 0; a < env.num_agents; a++) sum += env.rewards[a];
        total_reward += sum;

        // c_reset zeroes the counter, so the delta is only readable mid-episode
        if (!last){
            float collected = env.tot_apple_collected - prev_apples;
            if (fabsf(sum - collected) > 1e-3f){
                if (failures < 5) printf("  FAIL t=%d sum(r)=%.4f apples=%.0f\n", env.tick, sum, collected);
                failures++;
            }
            prev_apples = env.tot_apple_collected;
        } else {
            prev_apples = 0.0f;
        }

        if (shared_rewards){
            for (int a = 1; a < env.num_agents; a++){
                if (fabsf(env.rewards[a] - env.rewards[0]) > 1e-6f){
                    if (failures < 5) printf("  FAIL t=%d unequal share a0=%.4f a%d=%.4f\n", env.tick, env.rewards[0], a, env.rewards[a]);
                    failures++;
                }
            }
        }
    }

    printf("shared_rewards=%d: %d failures over %d episodes\n", shared_rewards, failures, num_episodes);
    printf("  logged mean episode_return = %.3f (n=%.0f)\n", env.log.episode_return/env.log.n, env.log.n);
    printf("  sum of all rewards / (num_agents*episodes) = %.3f\n", total_reward/(env.num_agents*(float)num_episodes));

    c_close(&env);
    free(env.observations);
    free(env.rewards);
    free(env.terminals);
    free(env.actions);
}

int main() {
    // demo();
    test_rewards(false);
    test_rewards(true);
    // performance_test();
    return 0;
}