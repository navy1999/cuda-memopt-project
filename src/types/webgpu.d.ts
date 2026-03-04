/**
 * Minimal WebGPU type for navigator.gpu (not in default DOM lib yet).
 */
interface GPUAdapter {
  requestDevice(): Promise<GPUDevice>;
}
interface GPU {
  requestAdapter(): Promise<GPUAdapter | null>;
}
declare global {
  interface Navigator {
    gpu?: GPU;
  }
}
export {};
