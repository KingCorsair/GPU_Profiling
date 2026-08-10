
// expected types for Server.py
type LoadConfig = {
  endpoint: string;
  requestsPerSecond: number;
  durationSeconds: number;
};

type InferRequest = {
  image_b64: string;
  question: string;
};

type InferResponse = {
  answer: string;
}

type RequestResult = {
  sequence: number;
  scheduledAtMs: number;
  sentAtMs: number;
  completedAtMs: number;
  status: number | null;
  error: string | null;
};

// checks if config paramrs are valid
function validateConfig(config: LoadConfig): void {
  if (config.requestsPerSecond <= 0) {
    throw new Error("RequestsPerSecond must be greater than 0");
  } else if (config.durationSeconds <= 0) {
    throw new Error("Duration seconds must be greater than 0");
  } else if (config.endpoint.trim() == "") {
    throw new Error("endpoint must be nonempty");
  }
}

const payload: InferRequest = {
  image_b64: Buffer.from("fake image bytes").toString("base64"),
  question: "What is in this image?",
};

//sends a single request to the server and returns the result
async function sendOne(
  sequence: number,
  scheduledAtMs: number,
  config: LoadConfig,
  payload: InferRequest,
): Promise<RequestResult> {
  const sentAtMs = performance.now();
  let status: number | null = null;

  try {
    const response = await fetch(config.endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    status = response.status;

    if (!response.ok) {
      await response.text();
      const completedAtMs = performance.now();

      return {
        sequence,
        scheduledAtMs,
        sentAtMs,
        completedAtMs,
        status,
        error: `Request failed with status ${status}`,
      };
    }

    const responseBody = (await response.json()) as InferResponse;
    if (typeof responseBody.answer !== "string") {
      throw new Error("Response body is missing a string answer");
    }
    const completedAtMs = performance.now();

    return {
      sequence,
      scheduledAtMs,
      sentAtMs,
      completedAtMs,
      status,
      error: null,
    };
  } catch (caught: unknown) {
    return {
      sequence,
      scheduledAtMs,
      sentAtMs,
      completedAtMs: performance.now(),
      status,
      error: caught instanceof Error ? caught.message : String(caught),
    };
  }
}

function sleep(ms:number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
  const config: LoadConfig = {
    endpoint: "http://127.0.0.1:8000/infer",
    requestsPerSecond: 5,
    durationSeconds: 10,
  };

  const scheduledAtMs = performance.now();
  const result = await sendOne(
    0,
    scheduledAtMs,
    config,
    payload,
  );
  console.log(result);
}

await main();