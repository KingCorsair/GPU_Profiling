export type ArrivalMode = "uniform" | "poisson";
export type ArrivalConfig = {
  arrival?: ArrivalMode; arrivalSeed?: number; seed: number;
  measuredRequests: number; requestsPerSecond: number; durationSeconds: number;
};

/** Absent fields retain the original deterministic schedule in archived manifests. */
export function arrivalMode(config: Pick<ArrivalConfig, "arrival">): ArrivalMode {
  const mode = config.arrival === undefined ? "uniform" : config.arrival;
  if (mode !== "uniform" && mode !== "poisson") throw new Error("arrival must be uniform or poisson");
  return mode;
}

function validateSeed(seed: number): void {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error("arrival seed must be an integer from 0 through 4294967295");
}
export function effectiveArrivalSeed(config: Pick<ArrivalConfig, "arrival" | "arrivalSeed" | "seed">): number | null {
  if (arrivalMode(config) === "uniform") {
    if (config.arrivalSeed !== undefined) throw new Error("arrivalSeed is only supported for poisson arrivals");
    return null;
  }
  const seed = config.arrivalSeed === undefined ? config.seed : config.arrivalSeed;
  validateSeed(seed);
  return seed;
}

/** Independent, domain-separated RNG; workload shuffling never consumes this stream. */
function arrivalRandom(seed: number): () => number {
  let state = (seed ^ 0xa7715eed) >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    // Strictly inside (0, 1): neither zero gaps nor logarithmic infinities.
    return (((value ^ value >>> 14) >>> 0) + 0.5) / 4294967296;
  };
}

export function buildArrivalOffsets(config: ArrivalConfig): number[] {
  if (!Number.isSafeInteger(config.measuredRequests) || config.measuredRequests <= 0) throw new Error("measuredRequests must be a positive finite integer");
  if (!Number.isFinite(config.requestsPerSecond) || config.requestsPerSecond <= 0) throw new Error("requestsPerSecond must be positive and finite");
  const mode = arrivalMode(config), seed = effectiveArrivalSeed(config);
  if (mode === "uniform") {
    const offsets = Array.from({ length: config.measuredRequests }, (_, index) => index * 1000 / config.requestsPerSecond);
    if (offsets.some((offset) => !Number.isFinite(offset))) throw new Error("Arrival schedule exceeds finite clock range");
    return offsets;
  }
  const random = arrivalRandom(seed!);
  let offset = 0;
  return Array.from({ length: config.measuredRequests }, () => {
    const next = offset + -Math.log1p(-random()) * 1000 / config.requestsPerSecond;
    if (!Number.isFinite(next) || next <= offset) throw new Error("Arrival schedule exceeds clock precision or finite range");
    offset = next;
    return offset;
  });
}

/** Poisson trials stop arrivals at event N; their seeded observation window varies. */
export function plannedDurationSeconds(config: ArrivalConfig): number {
  if (arrivalMode(config) === "poisson") return buildArrivalOffsets(config).at(-1)! / 1000;
  effectiveArrivalSeed(config);
  if (!Number.isFinite(config.durationSeconds) || config.durationSeconds <= 0) throw new Error("durationSeconds must be positive and finite");
  return config.durationSeconds;
}
