#define HARVEST_DEBUG
#include "common_harvest.h"

static uint32_t action_rng = 12345;

static inline uint32_t next_action_rng(void){
    uint32_t x = action_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    action_rng = x;
    return x;
}

static void step_random(Env* env){
    for (int a = 0; a < env->num_agents; a++){
        env->agents[a].actions[0] = (float)(next_action_rng() % NUM_ACTIONS);
    }
    puf_step(env);
}

int main(int argc, char** argv){
    Ini ini = {0};
    puf_ini_load_env(&ini, "common_harvest", argc - 1, argv + 1);
    Dict* env_sec = puf_ini_section(&ini, "env", 0);

    Env env = {0};
    puf_init(&env, env_sec);
    action_rng += (uint32_t)dict_get(env_sec, "rng");

    obs_t* observations = calloc((size_t)env.num_agents*OBS_SIZE, sizeof(obs_t));
    float* actions = calloc((size_t)env.num_agents*NUM_ATNS, sizeof(float));
    float* rewards = calloc(env.num_agents, sizeof(float));
    float* terminals = calloc(env.num_agents, sizeof(float));
    for (int i = 0; i < env.num_agents; i++){
        env.agents[i].observations = observations + i*OBS_SIZE;
        env.agents[i].actions = actions + i*NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
    }
    puf_reset(&env);
    puf_render(&env);
    SetTargetFPS(60);

    printf("SPACE pause/resume | RIGHT step (paused, hold to repeat) | UP/DOWN speed | ESC quit\n");
    bool paused = true;
    float steps_per_sec = 4.0f;
    double accum = 0.0;
    while (!WindowShouldClose()){
        if (IsKeyPressed(KEY_SPACE)){
            paused = !paused;
            accum = 0.0;
            printf(paused ? "-- paused --\n" : "-- running at %.0f steps/s --\n", steps_per_sec);
        }
        if (IsKeyPressed(KEY_UP)) steps_per_sec = fminf(steps_per_sec*2.0f, 240.0f);
        if (IsKeyPressed(KEY_DOWN)) steps_per_sec = fmaxf(steps_per_sec/2.0f, 1.0f);

        if (paused){
            if (IsKeyPressed(KEY_RIGHT) || IsKeyPressedRepeat(KEY_RIGHT)) step_random(&env);
        } else {
            accum += GetFrameTime();
            while (accum >= 1.0/steps_per_sec){
                accum -= 1.0/steps_per_sec;
                step_random(&env);
            }
        }
        fflush(stdout);
        puf_render(&env);
    }

    puf_close(&env);
    free(observations);
    free(actions);
    free(rewards);
    free(terminals);
    puf_ini_free(&ini);
    return 0;
}
