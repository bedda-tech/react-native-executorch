import { LLMModule } from '../modules/natural_language_processing/LLMModule';
import type { LLMCapability, LLMModelName } from '../types/llm';
import type { ResourceSource } from '../types/common';

export type DownloadableModel = {
  modelName: LLMModelName;
  modelSource: ResourceSource;
  tokenizerSource: ResourceSource;
  tokenizerConfigSource: ResourceSource;
  capabilities?: readonly LLMCapability[];
};

/**
 * Download a model to the device cache without keeping the loaded instance.
 *
 * Wraps `LLMModule.fromModelName` to expose only the download concern.
 * Useful in onboarding flows where the download should complete before the
 * model is actually used.
 *
 * @param model - A model config object (e.g. `GEMMA4_E4B_QUANTIZED`).
 * @param onProgress - Called with a value in [0, 1] as bytes arrive.
 */
export async function downloadModel(
  model: DownloadableModel,
  onProgress: (progress: number) => void = () => {}
): Promise<void> {
  await LLMModule.fromModelName(model, onProgress);
}
