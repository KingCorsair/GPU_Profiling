import assert from "node:assert/strict";
import test from "node:test";
import { arrivalMode, effectiveArrivalSeed, buildArrivalOffsets, plannedDurationSeconds, type ArrivalConfig } from "./arrivals.js";

const base: ArrivalConfig = { seed: 7, measuredRequests: 12, requestsPerSecond: 100, durationSeconds: 0.12 };

test("legacy and explicit uniform arrivals retain their exact schedule and duration", () => {
  const expected = Array.from({ length: 12 }, (_, index) => index * 1000 / 100);
  assert.equal(arrivalMode(base), "uniform"); assert.equal(effectiveArrivalSeed(base), null);
  assert.deepEqual(buildArrivalOffsets(base), expected);
  assert.deepEqual(buildArrivalOffsets({ ...base, arrival: "uniform" }), expected);
  assert.equal(plannedDurationSeconds(base), 0.12);
});

test("Poisson gaps reproduce by arrival seed and stay independent of workload randomness", () => {
  const config: ArrivalConfig = { ...base, arrival: "poisson", arrivalSeed: 42 };
  const offsets = buildArrivalOffsets(config);
  assert.deepEqual(offsets, buildArrivalOffsets(config));
  assert.deepEqual(offsets, buildArrivalOffsets({ ...config, seed: 999 }));
  assert.notDeepEqual(offsets, buildArrivalOffsets({ ...config, arrivalSeed: 43 }));
  assert.equal(effectiveArrivalSeed({ ...base, arrival: "poisson" }), base.seed);
  assert.deepEqual(buildArrivalOffsets({ ...base, arrival: "poisson" }), buildArrivalOffsets({ ...base, arrival: "poisson", arrivalSeed: base.seed }));
  assert.ok(offsets[0]! > 0);
  assert.ok(offsets.every((value, index) => index === 0 || value > offsets[index - 1]!));
  assert.equal(plannedDurationSeconds(config), offsets.at(-1)! / 1000);
  assert.notEqual(plannedDurationSeconds(config), base.measuredRequests / base.requestsPerSecond);
  assert.deepEqual(offsets.slice(0, 4), buildArrivalOffsets({ ...config, measuredRequests: 4 }));
  const twiceRate = buildArrivalOffsets({ ...config, requestsPerSecond: 200 });
  offsets.forEach((offset, index) => assert.equal(twiceRate[index], offset / 2));
});

test("a large seeded Poisson schedule has exponential gaps near its declared mean", () => {
  const config: ArrivalConfig = { ...base, arrival: "poisson", arrivalSeed: 465, measuredRequests: 100000, requestsPerSecond: 4 };
  const offsets = buildArrivalOffsets(config);
  const gaps = offsets.map((offset, index) => offset - (index === 0 ? 0 : offsets[index - 1]!));
  const mean = offsets.at(-1)! / gaps.length;
  const variance = gaps.reduce((sum, gap) => sum + (gap - mean) ** 2, 0) / gaps.length;
  assert.ok(Math.abs(mean - 250) < 5, `seeded mean gap ${mean}ms differs from expected 250ms`);
  assert.ok(Math.abs(Math.sqrt(variance) / mean - 1) < 0.03, "exponential gaps should have a coefficient of variation near one");
  const shortFraction = gaps.filter((gap) => gap < 250).length / gaps.length;
  assert.ok(Math.abs(shortFraction - (1 - Math.exp(-1))) < 0.01);
});

test("arrival settings reject malformed seeds, modes, budgets and unrepresentable clocks", () => {
  for (const arrival of [null, "", "burst"]) assert.throws(() => arrivalMode({ arrival } as unknown as ArrivalConfig), /arrival must/);
  assert.throws(() => effectiveArrivalSeed({ ...base, arrivalSeed: 1 }), /only supported for poisson/);
  for (const arrivalSeed of [null, "1", -1, 0.5, NaN, Infinity, 4294967296]) {
    assert.throws(() => buildArrivalOffsets({ ...base, arrival: "poisson", arrivalSeed } as unknown as ArrivalConfig), /arrival seed/);
  }
  for (const measuredRequests of [0, -1, 1.5, NaN, Infinity]) assert.throws(() => buildArrivalOffsets({ ...base, measuredRequests }), /measuredRequests/);
  for (const requestsPerSecond of [0, -1, NaN, Infinity]) assert.throws(() => buildArrivalOffsets({ ...base, requestsPerSecond }), /requestsPerSecond/);
  assert.throws(() => buildArrivalOffsets({ ...base, arrival: "poisson", requestsPerSecond: Number.MIN_VALUE }), /clock precision or finite range/);
});
