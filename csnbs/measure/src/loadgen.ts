
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
};

// checks if config paramrs are valid
function validateConfig(config: LoadConfig): void {
  if (config.requestsPerSecond <= 0) {
    throw new Error("RequestsPerSecond must be greater than 0");
  } else if (config.durationSeconds <=0){
    throw new Error("Duration seconds must be greater than 0");
  } else if (config.endpoint.trim() == ""){
    throw new Error("endpoint must be nonempty");
  }
}

const payload: InferRequest = {
  image_b64: Buffer.from("fake image bytes").toString("base64"),
  question: "What is in this image?",
};

async function sendOne(
config: LoadConfig,
payload: InferRequest,
): Promise<InferResponse> {
  const response = await fetch(config.endpoint,
    {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
    });
    if(!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    return (await response.json()) as InferResponse
  }

  async function main(): Promise<void> {
  const config: LoadConfig = {
    endpoint: "http://127.0.0.1:8000/infer",
    requestsPerSecond: 5,
    durationSeconds: 10,
  };

  validateConfig(config);

  const result = await sendOne(config, payload);

  console.log(result.answer);
}

await main();