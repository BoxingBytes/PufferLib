#include "cleanup.h"
#define OBS_SIZE (NUM_OBS_CHANNELS*OBS_WINDOW*OBS_WINDOW)
#define NUM_ATNS 1
#define ACT_SIZES {NUM_ACTIONS}
#define OBS_TENSOR_T ByteTensor

#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->differentiate_other_agents_in_obs = dict_get(kwargs, "differentiate_other_agents_in_obs")->value;
    env->shared_rewards = dict_get(kwargs, "shared_rewards")->value;
    env->rng += (uint32_t)dict_get(kwargs, "rng")->value;  // my_vec_init preloads rng with the env index
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "time_to_growth", log->time_to_growth);
    dict_set(out, "mean_dirt_fraction", log->mean_dirt_fraction);
    dict_set(out, "clean_rate", log->clean_rate);
    dict_set(out, "clean_hit_rate", log->clean_hit_rate);
    dict_set(out, "sustainability", log->sustainability);
    dict_set(out, "efficiency", log->efficiency);
    dict_set(out, "zap_rate", log->zap_rate);
    dict_set(out, "hit_rate", log->hit_rate);
    dict_set(out, "zap_timing", log->zap_timing);
    dict_set(out, "hit_timing", log->hit_timing);
    dict_set(out, "equality", log->equality);
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
