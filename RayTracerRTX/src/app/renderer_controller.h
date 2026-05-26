#pragma once

#include "app_state.h"

void invalidateAccumulation(AppState& appState);
void requestRendererSceneRebuild(AppState& appState);
void markSceneEdited(AppState& appState, bool changed, bool rebuildScene = false);
void applyQualityMode(AppState& appState);
