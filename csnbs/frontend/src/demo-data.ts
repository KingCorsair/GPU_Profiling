// UI fixtures only. These values are invented, not measurements or predictions.
// Never import this module into evidence exports, campaign reports, or the accuracy join.
export const demoRates = [1, 2, 3, 4, 5] as const;
export const demoCategories = ['Description', 'Object presence', 'Fine attributes', 'OCR', 'Counting', 'Spatial reasoning'] as const;
export type DemoPath = 'reference' | 'proposed';
export type DemoMeasurements = { throughput: number[]; p50: number[]; p95: number[] };
export type DemoConfiguration = {
  tokens: number;
  accuracy: number;
  categories: number[];
  reference: DemoMeasurements;
  proposed: DemoMeasurements;
};

export const illustrativeStudy: { kind: 'illustrative'; configurations: DemoConfiguration[] } = {
  kind: 'illustrative',
  configurations: [
    { tokens: 576, accuracy: 82, categories: [91, 89, 82, 75, 77, 78],
      reference: { throughput: [1, 1.9, 2.3, 2.3, 2.2], p50: [510, 620, 2400, 6400, 11500], p95: [840, 1200, 5800, 14000, 24000] },
      proposed: { throughput: [1, 2, 2.9, 3.5, 3.5], p50: [490, 530, 690, 1700, 4100], p95: [810, 900, 1200, 3600, 8500] } },
    { tokens: 288, accuracy: 81, categories: [91, 88, 82, 72, 75, 78],
      reference: { throughput: [1, 2, 2.5, 2.6, 2.5], p50: [450, 520, 1600, 5100, 9400], p95: [780, 1030, 4300, 11500, 20300] },
      proposed: { throughput: [1, 2, 3, 3.8, 4.1], p50: [430, 450, 520, 1100, 2700], p95: [720, 780, 980, 2600, 6100] } },
    { tokens: 128, accuracy: 76, categories: [89, 87, 77, 60, 65, 78],
      reference: { throughput: [1, 2, 2.7, 2.9, 2.8], p50: [400, 450, 1100, 3900, 7600], p95: [700, 920, 3200, 8800, 16700] },
      proposed: { throughput: [1, 2, 3, 4, 4.6], p50: [380, 400, 460, 670, 1400], p95: [650, 710, 820, 1300, 3400] } },
    { tokens: 64, accuracy: 61, categories: [81, 77, 63, 32, 42, 71],
      reference: { throughput: [1, 2, 2.8, 3, 2.9], p50: [370, 420, 980, 3400, 6900], p95: [680, 890, 2900, 7900, 15300] },
      proposed: { throughput: [1, 2, 3, 4, 4.7], p50: [350, 370, 430, 590, 1200], p95: [620, 680, 780, 1150, 2900] } },
  ],
};
