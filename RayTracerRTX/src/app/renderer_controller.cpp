#include "renderer_controller.h"

void invalidateAccumulation(AppState& appState)
{
    appState.progressiveSamples = 0;
}

void requestRendererSceneRebuild(AppState& appState)
{
    appState.rendererSceneRebuildRequested = true;
}

void markSceneEdited(AppState& appState, const bool changed, const bool rebuildScene)
{
    if (!changed)
    {
        return;
    }

    invalidateAccumulation(appState);
    if (rebuildScene)
    {
        requestRendererSceneRebuild(appState);
    }
}

void pushUndoSnapshot(AppState& appState, const SceneState& scene)
{
    constexpr size_t maxUndoSnapshots = 32;
    appState.undoStack.push_back(scene);
    if (appState.undoStack.size() > maxUndoSnapshots)
    {
        appState.undoStack.erase(appState.undoStack.begin());
    }
}

bool undoLastSceneEdit(AppState& appState)
{
    if (appState.undoStack.empty())
    {
        return false;
    }

    appState.scene = appState.undoStack.back();
    appState.undoStack.pop_back();
    invalidateAccumulation(appState);
    requestRendererSceneRebuild(appState);
    return true;
}

void applySceneEditResult(AppState& appState, const SceneEditResult& result)
{
    if (!result.changed)
    {
        return;
    }

    invalidateAccumulation(appState);
    if (needsRendererSceneRebuild(result.dirty))
    {
        requestRendererSceneRebuild(appState);
    }
}

void applySceneEditResultWithUndo(AppState& appState, const SceneState& before, const SceneEditResult& result)
{
    if (result.changed)
    {
        pushUndoSnapshot(appState, before);
    }
    applySceneEditResult(appState, result);
}

void applyQualityMode(AppState& appState)
{
    appState.renderQuality = clampRenderQuality(appState.renderQuality);
    if (renderQualityUsesPathTracing(appState.renderQuality))
    {
        appState.renderMode = RenderModeProgressive;
        appState.denoiserEnabled = renderQualityUsesDenoiser(appState.renderQuality);
    }
    else
    {
        appState.renderMode = RenderModeRealtime;
        appState.denoiserEnabled = false;
    }
    invalidateAccumulation(appState);
}
