#pragma once

#include "app_state.h"
#include "scene_editor.h"

void invalidateAccumulation(AppState& appState);
void requestRendererSceneRebuild(AppState& appState);
void markSceneEdited(AppState& appState, bool changed, bool rebuildScene = false);
void pushUndoSnapshot(AppState& appState, const SceneState& scene);
bool undoLastSceneEdit(AppState& appState);
void applySceneEditResult(AppState& appState, const SceneEditResult& result);
void applySceneEditResultWithUndo(AppState& appState, const SceneState& before, const SceneEditResult& result);
void applyQualityMode(AppState& appState);
