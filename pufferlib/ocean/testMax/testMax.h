#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_GREEN = (Color){0, 187, 0, 255};
const Color PUFF_YELLOW = (Color){187, 187, 0, 255};

// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field 
} Log;

typedef struct {
    Log log;                     // Required field
    unsigned char* observations; // Required field. Ensure type matches in .py and .c
    int* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    int x;
    Color color;
} testMax;

void c_reset(testMax* env) {
    env->x = rand() %5; 
    env->color = PUFF_BACKGROUND;
    env->observations[0] = env->x; 
}

void c_step(testMax* env) {
    env->rewards[0] = 0;
    env->terminals[0] = 0; //1; // Episode ends after one step

    if (env->x == env->actions[0]) {
        env->rewards[0] = 1;
        env->log.score += 1;
        env->log.n += 1;
        env->color = PUFF_GREEN;
    } else {
        env->rewards[0] = -1;
        env->log.score -= 1;
        env->log.n += 1;
        env->color = PUFF_RED;
    }
}

void c_render(testMax* env) {
    if (!IsWindowReady()) {
        InitWindow(256, 256, "PufferLib testMax");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    char buf[64];
    snprintf(buf, sizeof(buf), "X: %d; A: %d, R: %.2f", env->x, env->actions[0], env->rewards[0]);
    DrawText(buf, 20, 20, 20, PUFF_WHITE);
    DrawRectangle(128-32, 128-32, 64, 64, env->color);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    EndDrawing();
}

void c_close(testMax* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
