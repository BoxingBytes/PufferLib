#include <time.h>
#include "cleanup.h"

void demo() {
    Env env = {
        .num_agents = 7,
        .differentiate_other_agents_in_obs = true,
        .shared_rewards = true,
        .rng = 42,
    };
    env.observations = (uint8_t*)calloc(env.num_agents*NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW, sizeof(uint8_t));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (float*)calloc(env.num_agents, sizeof(float));
    env.actions = (float*)calloc(env.num_agents, sizeof(float));

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
            else if (IsKeyDown(KEY_C))     action = ACTION_CLEAN;
            env.actions[human_agent_idx] = action;
        }

        c_step(&env);
        c_render(&env);
    }
    c_close(&env);

    free(env.observations);
    free(env.rewards);
    free(env.terminals);
    free(env.actions);
}

int main() {
    demo();
    return 0;
}
