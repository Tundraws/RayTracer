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
