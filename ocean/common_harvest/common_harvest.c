#include "common_harvest.h"

int main() {
    Env env = {
        .num_agents = 4,
        .obs_size = 11,
    };
    env.observations = (uint8_t*)calloc(env.num_agents*NUM_OBS_PLANES*env.obs_size*env.obs_size, sizeof(uint8_t));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));
    env.actions = (int32_t*)calloc(env.num_agents, sizeof(int32_t));
    env.masks = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));

    init(&env);
    c_reset(&env);
    c_render(&env);
    
    while (!WindowShouldClose()) {
        c_reset(&env);
        c_render(&env);
    }
    c_close(&env);

    free(env.observations);
    free(env.rewards);
    free(env.terminals);
    free(env.actions);
    free(env.masks);
    return 0;
}